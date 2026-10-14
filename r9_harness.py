"""R9 node-level intervention harness (HarnessPAI discipline, task-level design steps 3-4).

Evaluation-legal inputs ONLY -- everything the policy already receives in obs:
  - head RGB (720x720, RGBA) + head depth_linear, via RGBDFullResWrapper
  - robot_r1::cam_rel_poses  (7-dim [x,y,z,qw,qx,qy,qz] per camera, robot-base frame)
  - robot_r1::proprio       (61-dim raw; eef_left_pos=[17:20], eef_right_pos=[42:45], robot-local)

Modes (config file, polled live -- no policy restart needed to switch):
  observe   : telemetry only. Grasp/empty-close events, eef trace, perception dry-run
              (blobs found, uv, depth, back-projected xyz). NEVER edits actions.
  intervene : observe + node-level watchdog -> ONE aim per trigger (perceived toy ->
              pre-grasp pose via online Broyden-Jacobian servo) -> stand down, VLA
              closes the gripper and finishes grasp/place (proven at 0.333 baseline).

Discipline (per HarnessPAI借鉴与task-level-harness设计.md §3.3/§4):
  - trigger: stage dwell > dwell_trigger AND >= N empty closes since last progress
             AND neither gripper holding AND stage in window AND budget left
  - single aim, frozen target, no grid scan, no pose forcing, no stage forcing
  - bounded interventions per episode; after aim -> yield window (VLA full control)
"""
import json
import logging
import os
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None

# --- state layout (identical to b1k.shared.correction_rules) ------------------
BASE_IDX = slice(0, 3)
TRUNK_IDX = slice(3, 7)
LEFT_ARM_IDX = slice(7, 14)
LEFT_GRIPPER_IDX = 14
RIGHT_ARM_IDX = slice(15, 22)
RIGHT_GRIPPER_IDX = 22

# raw 61-dim proprio slices (PROPRIOCEPTION_INDICES["R1Pro"])
EEF_LEFT_SLICE = slice(17, 20)
EEF_RIGHT_SLICE = slice(42, 45)

OPEN_THRESHOLD = 0.90
CLOSED_THRESHOLD = -0.98
EMPTY_CLOSE_T = -0.88          # width below this after a close command = closed on nothing
HOLD_MIN_TRAVEL = 0.25         # finger travel needed to believe a payload

DEFAULT_CONFIG = {
    "mode": "observe",              # observe | intervene
    "tasks": [7],                   # task_ids the harness runs on (task 7 = picking_up_toys)
    "log_path": "/logs/r9_harness.jsonl",
    "steps_per_call": 20,           # = wrapper execute_in_n_steps
    # ---- watchdog trigger ----
    "dwell_trigger": 3000,          # sim steps without stage advance before eligible
    "empty_trigger": 2,             # empty closes since last progress needed
    "no_attempt_trigger": 1500,     # OR: no close commanded for this long while stuck
                                    # (catches the "gave up / never attempts" mode observed on 301)
    "stage_min": 2,
    "stage_max": 13,
    "max_interventions": 8,
    "perception_fail_limit": 4,     # after this many failed aims, stop trying this episode
    "blacklist_r": 0.35,            # after an aim yields no grasp, exclude that spot
    # ---- aim / servo ----
    "servo_budget": 60,            # re-plans (x20 steps) max per aim
    "servo_gain": 0.8,
    "servo_max_dq": 0.08,
    "servo_tol": 0.15,             # arrive zone: palm->toy-center; fingers reach ~0.1m further
    "stall_window": 15,            # re-plans; abort when err improves < stall_improve
    "stall_improve": 0.035,        # over that window (int301-b: 3 stalled aims
                                   # burned 3600 steps; the 4th, converging one,
                                   # then ran out of episode)
    "explore_amp": 0.05,
    "arm_side_margin": 0.12,       # target |y| beyond this -> use that side's arm
    "close_calls": 10,             # harness-driven close window after arrival (x20 steps)
    "close_retries": 2,            # empty close -> descend target & re-grasp, same aim
    "reapproach_budget": 15,       # re-plans for each descend-retry servo leg
    "descend_step": 0.06,          # per-retry aim-target descent (m)
                                   # (int301-a/b: VLA reliably fails to close even with
                                   #  fingers on the toy -- "gave up" mode)
    "pre_grasp_offset": [0.0, 0.0, 0.03],   # palm aimed this far above toy center, robot-local
    "hover_adapt": True,           # empty close at arrival -> descend hover next aim
    "hover_adapt_step": 0.04,      # per-empty-close descent
    "hover_min": -0.02,            # floor for the adaptive hover height
    "reach_h_min": 0.30,           # horizontal (xy) distance from base origin
    "reach_h_max": 1.10,           # arm cannot reach past this; base must move (out of scope v0)
    "z_min": 0.35,                  # empirical toy zone (bed 0.50 .. tabletop 0.83);
    "z_max": 0.95,                  # floor blobs (0.0-0.3) and head-height furniture
                                   # (1.0+) are unreachable or not toys
    "far_cooldown": 600,           # steps to wait when the perceived target is out of reach
    # ---- yield ----
    "yield_steps": 1500,           # after aim: hands-off window for the VLA
    "yield_fail_steps": 800,       # shorter hand-back when the aim clearly failed
    "yield_hold_refresh": 3000,    # extend while carrying (VLA transports+places)
    # ---- perception ----
    "intrinsics": None,            # {"fx","fy","cx","cy"} -- head cam calibrated: fx=306, fy=800, cx=cy=360
    "cam_pose_idx": 2,             # index of head camera in cam_rel_poses (7-dim blocks; order l,r,head)
    "quat_order": "xyzw",          # omnigibson transform_utils convention (validated empirically)
    "depth_mode": "ray",           # depth_linear is Euclidean ray length (validated: tennis ball 2% err)
    "flip_v": True,                # v axis points down in image; y up in camera frame
    # band priority order (validated vs sim ground truth on 301): [25,40] median 0.12m;
    # red [0,15] is mostly furniture (median 0.44m) -> last resort with tight size caps
    "hsv_bands_deg": [[25, 40], [40, 85], [15, 25], [0, 15]],
    "s_min": 0.40,
    "v_min": 0.25,
    "depth_min": 0.30,
    "depth_max": 2.2,
    "blob_min_px": 150,
    "blob_max_px": 5000,           # exclude table/floor-scale surfaces
    "log_every": 1,                # write a telemetry row every N calls
}


def _rgb_to_hsv(rgb):
    """rgb float [0,255] (H,W,3) -> hsv with h in [0,180] (OpenCV scale), s,v in [0,1]."""
    r = rgb[..., 0] / 255.0
    g = rgb[..., 1] / 255.0
    b = rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn + 1e-12
    h = np.zeros_like(mx)
    m = (mx == r) & (d > 1e-9)
    h[m] = np.mod((g - b)[m] / d[m], 6.0)
    m = (mx == g) & (d > 1e-9)
    h[m] = (b - r)[m] / d[m] + 2.0
    m = (mx == b) & (d > 1e-9)
    h[m] = (r - g)[m] / d[m] + 4.0
    h = h * 30.0
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h, s, mx], axis=-1)


def _quat2rot(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class EefController:
    """Online Broyden-Jacobian position servo (copied from champion harness critic.py,
    validated: eef error 0.472 -> 0.024 on instance 301)."""

    def __init__(self, n_joints=7, explore_amp=0.05):
        self.n = n_joints
        self.J = np.zeros((3, n_joints))
        self.known = np.zeros(n_joints, dtype=bool)
        self.last_q = None
        self.last_e = None
        self.updates = 0
        self.explore_amp = explore_amp

    def reset(self):
        self.J = np.zeros((3, self.n))
        self.known[:] = False
        self.last_q = None
        self.last_e = None
        self.updates = 0

    def observe(self, q, e):
        q = np.asarray(q, dtype=float).reshape(-1)
        e = np.asarray(e, dtype=float).reshape(-1)
        if q.shape[0] != self.n or e.shape[0] != 3:
            self.last_q, self.last_e = None, None
            return
        if self.last_q is not None:
            dq = q - self.last_q
            de = e - self.last_e
            moved = np.flatnonzero(np.abs(dq) > 1e-4)
            if moved.size == 1 and abs(dq[moved[0]]) > 1e-3:
                i = int(moved[0])
                self.J[:, i] = de / dq[i]
                self.known[i] = True
                self.updates += 1
            elif moved.size > 1:
                nq = float(np.linalg.norm(dq))
                if nq > 1e-3:
                    pred = self.J @ dq
                    resid = de - pred
                    upd = np.outer(resid, dq) / (nq * nq)
                    if np.linalg.norm(upd) < 5.0:
                        self.J += upd
        self.last_q = q.copy()
        self.last_e = e.copy()

    def explore(self, q, amp=None):
        amp = self.explore_amp if amp is None else amp
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.n:
            return None
        todo = np.flatnonzero(~self.known)
        if todo.size == 0:
            return None
        out = q.copy()
        out[int(todo[0])] += amp
        return out

    @property
    def ready(self):
        return bool(self.known.all())

    def command(self, q, e, target, gain=0.7, max_dq=0.10, tol=0.03):
        if not self.ready:
            return None
        q = np.asarray(q, dtype=float).reshape(-1)
        e = np.asarray(e, dtype=float).reshape(-1)
        err = np.asarray(target, dtype=float) - e
        dist = float(np.linalg.norm(err))
        if dist <= tol:
            return q
        try:
            Jp = np.linalg.pinv(self.J)
        except Exception:
            return None
        dq = Jp @ err
        n = float(np.linalg.norm(dq))
        if n < 1e-6:
            return None
        if n > max_dq:
            dq = dq * (max_dq / n)
        return q + dq * gain


class R9Harness:
    def __init__(self):
        self.cfg = dict(DEFAULT_CONFIG)
        self.cfg["flip_v"] = True
        self.config_path = os.environ.get("B1K_R9_CONFIG", "/logs/r9_harness_config.json")
        self._cfg_mtime = None
        self._user_cfg = {}
        self.task_profile = {}
        self._load_config()
        self._reset_episode(-1)

    # ------------------------------------------------------------- config
    def _rebuild_cfg(self):
        """Effective cfg = DEFAULT <- base config file <- frozen per-task profile.
        Rebuilt fresh each load/reset so per-task overrides never leak across
        episodes of different tasks."""
        self.cfg = dict(DEFAULT_CONFIG)
        self.cfg["flip_v"] = True
        self.cfg.update(self._user_cfg or {})
        self.cfg.update(self.task_profile)

    def _load_config(self):
        try:
            st = os.stat(self.config_path)
            if self._cfg_mtime != st.st_mtime:
                with open(self.config_path) as fh:
                    self._user_cfg = json.load(fh)
                self._cfg_mtime = st.st_mtime
                logger.info(f"R9H: config loaded/reloaded (file mode={self._user_cfg.get('mode')} "
                            f"intrinsics={self._user_cfg.get('intrinsics')})")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"R9H: config load failed: {e}")
        self._rebuild_cfg()

    def _load_profile(self, task_id):
        """Frozen per-task profile: knobs the cross-episode tuner owns. Layered
        DEFAULT <- base config file <- profile_task{N}.json (highest)."""
        try:
            d = os.path.dirname(self.config_path) or "/logs"
            p = os.path.join(d, f"profile_task{int(task_id)}.json")
            with open(p) as fh:
                prof = json.load(fh)
            self.task_profile = {k: v for k, v in prof.items() if not k.startswith("_")}
            logger.info(f"R9H: profile task{task_id} loaded: {sorted(self.task_profile)}")
        except FileNotFoundError:
            self.task_profile = {}
        except Exception as e:
            logger.warning(f"R9H: profile load failed: {e}")
            self.task_profile = {}

    # ------------------------------------------------------------- episode
    def reset_episode(self, task_id=None):
        """Public: called by the wrapper at each rollout reset."""
        if task_id is None:
            task_id = self.task_id
        self._reset_episode(task_id)

    def _reset_episode(self, task_id):
        self.task_id = task_id
        if task_id is not None and int(task_id) >= 0:
            self._load_profile(task_id)
        self._rebuild_cfg()
        self.hover_z = float(self.cfg["pre_grasp_offset"][2])   # adaptive within episode
        self.stage = 0
        self.best_stage = 0
        self.dwell = 0
        self.calls = 0
        self.step = 0
        self.n_interventions = 0
        self.perception_fails = 0
        self.far_cooldown_left = 0
        self.empty_since_progress = 0
        self.no_attempt_steps = 0
        self.target_blacklist = []
        self.aim_got_grasp = False
        self.phase = "idle"          # idle | aim | close | yield
        self.aim_target = None
        self.aim_arm = "l"
        self.aim_budget = 0
        self.aim_dist0 = None
        self.aim_best_err = None
        self.aim_err_hist = []
        self.close_left = 0
        self.close_retries = 0
        self.close_hist = []
        self.frozen_state = None
        self.yield_left = 0
        self.pending_cmd = {"l": 0.0, "r": 0.0}
        self.grip = {
            "l": {"state": "idle", "w0": None, "steps": 0},
            "r": {"state": "idle", "w0": None, "steps": 0},
        }
        self.holding = False
        self.ctrl = {
            "l": EefController(7, self.cfg["explore_amp"]),
            "r": EefController(7, self.cfg["explore_amp"]),
        }
        self._log_fh = None
        self._t0 = time.time()

    # ------------------------------------------------------------- logging
    def _log(self, row):
        path = self.cfg.get("log_path") or ""
        if not path:
            return
        row["wall"] = round(time.time() - self._t0, 2)
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------- helpers
    def _extract_eef(self, obs):
        el = er = None
        try:
            arr = np.asarray(obs.get("robot_r1::proprio"), dtype=float).reshape(-1)
            if arr.shape[0] >= 45:
                el = arr[EEF_LEFT_SLICE].copy()
                er = arr[EEF_RIGHT_SLICE].copy()
        except Exception:
            pass
        return el, er

    def _head_keys(self, obs):
        for k in obs.keys():
            if k.endswith("::rgb") and "zed" in k:
                rgb_key = k
                depth_key = k[: -len("::rgb")] + "::depth_linear"
                if depth_key in obs:
                    return rgb_key, depth_key
                return rgb_key, None
        return None, None

    def _cam_pose7(self, obs):
        try:
            crp = np.asarray(obs.get("robot_r1::cam_rel_poses"), dtype=float).reshape(-1)
            i = int(self.cfg["cam_pose_idx"])
            if crp.shape[0] >= 7 * (i + 1):
                return crp[7 * i: 7 * i + 7]
        except Exception:
            pass
        return None

    # ------------------------------------------------------------- perception
    def _perceive(self, obs):
        """Return (target_xyz_local, info) or (None, info). Never raises."""
        info = {"blobs": []}
        try:
            rgb_key, depth_key = self._head_keys(obs)
            if rgb_key is None or depth_key is None:
                info["err"] = "no head keys"
                return None, info
            rgb = np.asarray(obs[rgb_key], dtype=float)
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                info["err"] = "bad rgb shape"
                return None, info
            rgb = rgb[..., :3]
            if rgb.max() <= 1.5:
                rgb = rgb * 255.0
            depth = np.asarray(obs[depth_key], dtype=float).squeeze()
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.shape != rgb.shape[:2]:
                info["err"] = f"depth shape {depth.shape} vs rgb {rgb.shape[:2]}"
                return None, info
            if ndimage is None:
                info["err"] = "no scipy"
                return None, info
            hsv = _rgb_to_hsv(rgb)
            s = hsv[..., 1]
            v = hsv[..., 2]
            base_mask = (s >= self.cfg["s_min"]) & (v >= self.cfg["v_min"])
            dmin, dmax = self.cfg["depth_min"], self.cfg["depth_max"]
            bmin, bmax = self.cfg["blob_min_px"], self.cfg["blob_max_px"]
            best = None
            # bands are in PRIORITY order: first band yielding a valid blob wins
            for h0, h1 in self.cfg["hsv_bands_deg"]:
                mask = base_mask & (hsv[..., 0] >= h0) & (hsv[..., 0] <= h1)
                if int(mask.sum()) < bmin:
                    continue
                lbl, n = ndimage.label(mask)
                if n <= 0:
                    continue
                sizes = ndimage.sum(mask, lbl, range(1, n + 1))
                order = np.argsort(sizes)[::-1][:5]
                for li in order:
                    comp = lbl == (li + 1)
                    px = int(comp.sum())
                    if px < bmin or px > bmax:
                        continue
                    rows_, cols_ = np.nonzero(comp)
                    # ---- shape/colour gates (v6): 12 aims, 0 at real toys ----
                    # warm-band blobs were table aprons, shelf wood, bed frames.
                    # Toys are COMPACT and moderately saturated; furniture wood
                    # is elongated (edges) or heavily saturated.
                    bw = int(cols_.max() - cols_.min() + 1)
                    bh = int(rows_.max() - rows_.min() + 1)
                    aspect = max(bw, bh) / max(1, min(bw, bh))
                    if aspect > float(self.cfg.get("blob_max_aspect", 3.0)):
                        continue
                    fill = px / float(bw * bh)
                    if fill < float(self.cfg.get("blob_min_fill", 0.30)):
                        continue
                    sm = float(np.median(s[comp]))
                    if sm > float(self.cfg.get("s_max", 0.55)):
                        continue
                    dd = depth[comp]
                    dd = dd[(dd > dmin) & (dd < dmax)]
                    if dd.size < 30:
                        continue
                    dm = float(np.median(dd))
                    u = float(np.median(cols_))
                    vv = float(np.median(rows_))
                    cand = {"band": [h0, h1], "u": round(u, 1), "v": round(vv, 1),
                            "d": round(dm, 3), "px": px, "sm": round(sm, 3),
                            "ar": round(aspect, 2)}
                    info["blobs"].append(cand)
                    if best is None or px > best["px"]:
                        best = cand
                if best is not None:
                    break  # first band with a valid blob decides the target
            if not info["blobs"]:
                info["err"] = "no_blob"
                return None, info
            # back-project every candidate blob; keep the reachable ones and pick
            # the one nearest to the closest end-effector (actionable target)
            intr = self.cfg.get("intrinsics")
            if not intr:
                info["err"] = "no_intr"
                return None, info
            cam7 = self._cam_pose7(obs)
            if cam7 is None:
                info["err"] = "no cam pose"
                return None, info
            fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
            t = cam7[:3]
            q = cam7[3:7]
            if self.cfg.get("quat_order") == "xyzw":
                q = np.array([q[3], q[0], q[1], q[2]])
            R = _quat2rot(q)
            ray_mode = self.cfg.get("depth_mode", "ray") == "ray"

            cands = []
            far = 0
            bl = self.target_blacklist
            bl_r = float(self.cfg.get("blacklist_r", 0.35))
            el, er = self._extract_eef(obs)
            eefs = [e for e in (el, er) if e is not None]
            eef_min = float(self.cfg.get("eef_min_dist", 0.25))
            for b in info["blobs"]:
                d = b["d"]
                tx = (b["u"] - cx) / fx
                ty = (cy - b["v"]) / fy if self.cfg.get("flip_v") else (b["v"] - cy) / fy
                if ray_mode:
                    norm = float(np.sqrt(1.0 + tx * tx + ty * ty))
                    p_cam = np.array([tx * d / norm, ty * d / norm, -d / norm])
                else:
                    p_cam = np.array([tx * d, ty * d, -d])
                p_loc = R @ p_cam + t
                # anti-"stuck-against" gate: when the VLA is wedged against
                # furniture, the nearest warm blob IS that furniture. Reject
                # candidates within eef_min_dist of either EEF.
                if eefs and eef_min > 0:
                    if any(float(np.linalg.norm(p_loc - e)) < eef_min for e in eefs):
                        far += 1
                        continue
                # reachability on the HORIZONTAL plane (the arm works above the
                # floor; a 3D norm from the base origin misjudges table targets)
                reach_h = float(np.linalg.norm(p_loc[:2]))
                if not (self.cfg["reach_h_min"] <= reach_h <= self.cfg["reach_h_max"]):
                    far += 1
                    continue
                if not (self.cfg["z_min"] <= p_loc[2] <= self.cfg["z_max"]):
                    far += 1
                    continue
                if bl and any(float(np.linalg.norm(p_loc - bp)) < bl_r for bp in bl):
                    far += 1
                    continue
                cands.append((p_loc, b))

            if not cands and bl:
                # every candidate is blacklisted: retry the OLDEST blacklisted
                # spot once (pop it) rather than doing nothing forever
                self.target_blacklist = bl[1:]
                row_note = "blacklist_retry"
                info["note"] = row_note

            if not cands:
                # distinguish "nothing detected" (burn a fail credit) from
                # "detected but out of reach" (cooldown only, the robot may
                # navigate closer and the target becomes actionable)
                info["err"] = "target_far" if far else "no_valid_blob"
                info["n_far"] = far
                return None, info

            # NOTE: EEF proximity already filtered above (anti stuck-against);
            # rank survivors by closeness to the nearest EEF (actionable first)
            if eefs:
                cands.sort(key=lambda cb: min(float(np.linalg.norm(cb[0] - e)) for e in eefs))
            else:
                cands.sort(key=lambda cb: -cb[1]["px"])
            p_loc, b = cands[0]
            info["target"] = [round(float(x), 3) for x in p_loc]
            info["target_blob"] = b
            info["reach_h"] = round(float(np.linalg.norm(p_loc[:2])), 3)
            return p_loc, info
        except Exception as e:
            info["err"] = repr(e)
            return None, info

    # ------------------------------------------------------------- gripper events
    def _update_gripper_events(self, lw, rw, row):
        events = []
        for side, w in (("l", lw), ("r", rw)):
            g = self.grip[side]
            cmd = self.pending_cmd[side]
            if cmd < -0.85:
                if g["w0"] is None:
                    g["w0"] = w
                g["steps"] += 1
                travel = float(g["w0"]) - w
                if w < EMPTY_CLOSE_T:
                    if g["state"] != "empty":
                        g["state"] = "empty"
                        self.empty_since_progress += 1
                        events.append({"evt": "empty_close", "side": side, "w": round(w, 3)})
                elif travel >= HOLD_MIN_TRAVEL and g["steps"] >= 2 and w < 0.7:
                    if g["state"] != "holding":
                        g["state"] = "holding"
                        events.append({"evt": "grasp_confirm", "side": side,
                                       "w": round(w, 3), "travel": round(travel, 3)})
                        if self.phase in ("aim", "close", "yield"):
                            self.aim_got_grasp = True
            else:
                if g["state"] == "holding":
                    events.append({"evt": "released", "side": side, "w": round(w, 3)})
                if g["state"] == "empty" and cmd >= -0.85:
                    g["state"] = "idle"
                g["state"] = "idle" if cmd >= -0.85 else g["state"]
                g["w0"] = None
                g["steps"] = 0
        self.holding = any(g["state"] == "holding" for g in self.grip.values())
        if events:
            row["gevents"] = events
        return events

    # ------------------------------------------------------------- main hook
    def __call__(self, task_id, stage, state, actions, obs=None, step_count=None):
        """Called at every re-plan (every ~20 sim steps). Returns possibly-edited actions."""
        try:
            if int(task_id) not in [int(t) for t in self.cfg["tasks"]]:
                return actions
        except Exception:
            return actions
        task_id = int(task_id)
        if task_id != self.task_id:
            self._reset_episode(task_id)
        self._load_config()
        self.calls += 1
        self.step = int(step_count) if step_count is not None else self.calls * 20
        self.stage = int(stage)

        st = np.asarray(state, dtype=float).reshape(-1)
        act = np.asarray(actions, dtype=float)
        lw = float(st[LEFT_GRIPPER_IDX])
        rw = float(st[RIGHT_GRIPPER_IDX])

        el, er = self._extract_eef(obs)

        # 1. jacobian observers always learn from real motion
        if el is not None:
            self.ctrl["l"].observe(st[LEFT_ARM_IDX], el)
        if er is not None:
            self.ctrl["r"].observe(st[RIGHT_ARM_IDX], er)

        # 2. gripper events from previous chunk's command vs current width
        row = {
            "step": self.step,
            "stage": int(stage),
            "best": self.best_stage,
            "dwell": self.dwell,
            "lw": round(lw, 3),
            "rw": round(rw, 3),
            "holding": self.holding,
            "empty_sp": self.empty_since_progress,
            "n_int": self.n_interventions,
            "phase": self.phase,
            "jkn_l": int(self.ctrl["l"].known.sum()),
            "jkn_r": int(self.ctrl["r"].known.sum()),
        }
        if el is not None:
            row["el"] = [round(float(x), 3) for x in el]
        if er is not None:
            row["er"] = [round(float(x), 3) for x in er]
        self._update_gripper_events(lw, rw, row)

        # "gave up" detector: how long since the VLA last commanded ANY close.
        # (pending_cmd still holds the PREVIOUS chunk's command at this point)
        if self.pending_cmd["l"] < -0.85 or self.pending_cmd["r"] < -0.85:
            self.no_attempt_steps = 0
        else:
            self.no_attempt_steps += self.cfg["steps_per_call"]

        # 3. progress tracking
        if self.stage > self.best_stage:
            self.best_stage = self.stage
            self.dwell = 0
            self.empty_since_progress = 0
            self.no_attempt_steps = 0
            row["evt"] = "stage_advance"
        else:
            self.dwell += self.cfg["steps_per_call"]

        # 4. dispatch (per-task mode override: e.g. {"7": "intervene", "3": "observe"}).
        # JSON object keys are ALWAYS strings, but task_id arrives as int ->
        # normalize with str() or the lookup silently misses and falls back to
        # the global mode (this exact bug contaminated the plates-301 run).
        eff_mode = self.cfg.get("mode_by_task", {}).get(
            str(task_id), self.cfg.get("mode_by_task", {}).get(task_id, self.cfg.get("mode")))
        row["mode"] = eff_mode
        out = None
        if eff_mode == "intervene":
            try:
                out = self._maybe_intervene(st, act, el, er, obs, row)
            except Exception:
                logger.exception("R9H: intervene failed; VLA passthrough")
                out = None
        elif self.calls % 5 == 0 or row.get("evt") or row.get("gevents"):
            # observe mode: perception dry-run every 5th call
            if obs is not None:
                _tgt, pinfo = self._perceive(obs)
                if pinfo.get("blobs"):
                    row["blobs"] = pinfo["blobs"][:3]
                if pinfo.get("target"):
                    row["ptarget"] = pinfo["target"]

        if out is not None:
            row["intervened"] = True
        # record the EFFECTIVE close intent (harness edits included) for the next
        # call's event check -- the harness-driven close must register, otherwise
        # holding/grasp detection never fires
        src = out if out is not None else act
        try:
            self.pending_cmd["l"] = float(np.min(src[:, LEFT_GRIPPER_IDX]))
            self.pending_cmd["r"] = float(np.min(src[:, RIGHT_GRIPPER_IDX]))
        except Exception:
            pass
        if self.calls % int(self.cfg.get("log_every", 1)) == 0 or row.get("evt") or row.get("gevents") or out is not None:
            self._log(row)
        return out if out is not None else actions

    # ------------------------------------------------------------- intervention
    def _maybe_intervene(self, st, act, el, er, obs, row):
        if self.phase == "yield":
            if self.holding:
                self.yield_left = max(self.yield_left, self.cfg["yield_hold_refresh"])
            self.yield_left -= self.cfg["steps_per_call"]
            if self.yield_left <= 0:
                self.phase = "idle"
                row["evt"] = "yield_end"
                if not self.aim_got_grasp and self.aim_target is not None:
                    # the VLA did not even try to close at our aim spot -> the
                    # target was wrong or unreachable; exclude it (round-robin
                    # through other candidates on the next trigger instead of
                    # looping on the same dead target)
                    self.target_blacklist.append(np.array(self.aim_target, dtype=float))
                    row["blacklisted"] = [round(float(x), 2) for x in self.aim_target]
                    logger.info(f"R9H: yield ended WITHOUT grasp -> blacklist {np.round(self.aim_target, 2).tolist()}")
                self.aim_got_grasp = False
            return None

        if self.phase == "close":
            return self._close_step(st, act, el, er, row)

        if self.phase == "aim":
            return self._aim_step(st, act, el, er, row)

        # idle -> check trigger. Two stuck modes (both observed on 301):
        #   a) repeated failed grasps: empty closes accumulate
        #   b) "gave up": grippers never commanded closed for a long stretch
        stuck_evidence = (
            self.empty_since_progress >= self.cfg["empty_trigger"]
            or self.no_attempt_steps >= self.cfg["no_attempt_trigger"]
        )
        trig = (
            self.dwell >= self.cfg["dwell_trigger"]
            and stuck_evidence
            and not self.holding
            and self.cfg["stage_min"] <= self.stage <= self.cfg["stage_max"]
            and self.n_interventions < self.cfg["max_interventions"]
            and self.perception_fails < self.cfg["perception_fail_limit"]
            and self.far_cooldown_left <= 0
        )
        if not trig:
            if self.far_cooldown_left > 0:
                self.far_cooldown_left -= self.cfg["steps_per_call"]
            return None

        tgt, pinfo = self._perceive(obs)
        row["blobs"] = pinfo.get("blobs", [])[:3]
        if tgt is None:
            err = pinfo.get("err", "")
            if err == "target_far":
                # perceived but out of arm reach: cool down, do NOT burn a fail
                # credit -- the VLA may navigate closer and make it actionable
                self.far_cooldown_left = int(self.cfg["far_cooldown"])
                row["evt"] = "target_far_cooldown"
                row["perr"] = err
                logger.info(f"R9H: target out of reach (n_far={pinfo.get('n_far')}) -> cooldown {self.cfg['far_cooldown']} steps")
            else:
                self.perception_fails += 1
                # no_blob burns a credit but MUST also cool down: int301-d
                # burned all 4 credits in 60 steps while the head camera was
                # simply not pointed at any toy, then sat dormant for 6880
                # steps of end-game stuck. Retry cooldown spreads the budget
                # across the whole stuck interval instead.
                self.far_cooldown_left = int(self.cfg.get("perception_retry_cooldown", 600))
                row["evt"] = "perception_fail"
                row["perr"] = err
                logger.info(f"R9H: perception fail #{self.perception_fails}: {err} "
                            f"-> retry in {self.cfg.get('perception_retry_cooldown', 600)} steps")
            return None

        off = np.asarray(self.cfg["pre_grasp_offset"], dtype=float).copy()
        off[2] = self.hover_z    # loop-level adaptation: descends after empty closes
        self.aim_target = np.asarray(tgt, dtype=float) + off
        dl = float(np.linalg.norm(np.asarray(el) - self.aim_target)) if el is not None else 1e9
        dr = float(np.linalg.norm(np.asarray(er) - self.aim_target)) if er is not None else 1e9
        # prefer the arm on the target's side: cross-body reaches stall early
        # (int301-b aim#3: left arm to y=-0.05 at horiz 1.08 dead-ended)
        margin = float(self.cfg.get("arm_side_margin", 0.12))
        ty = float(self.aim_target[1])
        if ty > margin:
            self.aim_arm = "l"
        elif ty < -margin:
            self.aim_arm = "r"
        else:
            self.aim_arm = "l" if dl <= dr else "r"
        self.aim_budget = int(self.cfg["servo_budget"])
        self.aim_best_err = None
        self.aim_err_hist = []
        self.close_retries = 0
        self.aim_dist0 = min(dl, dr)
        self.phase = "aim"
        self.n_interventions += 1
        row["evt"] = "aim_start"
        row["arm"] = self.aim_arm
        row["aim_target"] = [round(float(x), 3) for x in self.aim_target]
        logger.info(
            f"R9H: aim #{self.n_interventions} start | arm={self.aim_arm} "
            f"target={np.round(self.aim_target, 3).tolist()} dist0={self.aim_dist0:.3f} "
            f"dwell={self.dwell} empty_sp={self.empty_since_progress} step={self.step}"
        )
        return self._aim_step(st, act, el, er, row)

    def _aim_step(self, st, act, el, er, row):
        e = el if self.aim_arm == "l" else er
        ctrl = self.ctrl[self.aim_arm]
        arm_slice = LEFT_ARM_IDX if self.aim_arm == "l" else RIGHT_ARM_IDX
        if e is None:
            self.phase = "idle"
            row["evt"] = "aim_abort_noeef"
            return None
        if self.aim_budget <= 0:
            self.phase = "yield"
            self.yield_left = int(self.cfg["yield_fail_steps"])
            row["evt"] = "aim_budget_out"
            logger.info("R9H: aim budget out -> short yield")
            return None
        err = float(np.linalg.norm(self.aim_target - np.asarray(e)))
        # stall watchdog (windowed): abort when the servo has not improved by
        # stall_improve over the last stall_window re-plans. Windowed (not
        # vs-best) so slow-but-real convergence (~0.002/call) is not killed.
        self.aim_best_err = err if self.aim_best_err is None else min(self.aim_best_err, err)
        self.aim_err_hist.append(err)
        win = int(self.cfg["stall_window"])
        stalled = False
        if len(self.aim_err_hist) > win:
            stalled = (self.aim_err_hist[-1 - win] - err) < float(self.cfg["stall_improve"])
        if stalled:
            near = float(self.cfg["servo_tol"]) + 0.05
            if err <= near:
                # stalled just outside arrive tolerance: fingers may still be on
                # the toy -- attempt the bounded close instead of walking away
                self.phase = "close"
                self.close_left = int(self.cfg["close_calls"])
                self.close_hist = []
                self.frozen_state = st.copy()
                row["evt"] = "aim_stall_close_try"
                logger.info(f"R9H: aim stalled near target (err {err:.3f}) -> close try")
                return None
            self.phase = "yield"
            self.yield_left = int(self.cfg["yield_fail_steps"])
            row["evt"] = "aim_stall_abort"
            row["aim_err"] = round(err, 3)
            logger.info(f"R9H: aim stalled (err {err:.3f}, best {self.aim_best_err:.3f}, "
                        f"budget left {self.aim_budget}) -> short yield")
            return None
        if err <= self.cfg["servo_tol"]:
            self.phase = "close"
            self.close_left = int(self.cfg["close_calls"])
            self.close_hist = []
            self.frozen_state = st.copy()
            row["evt"] = "aim_arrived"
            logger.info(f"R9H: aim arrived (err {err:.3f}, d0 {self.aim_dist0:.3f}) -> "
                        f"harness close phase ({self.close_left} calls)")
            return None
        q = st[arm_slice].astype(float)
        if not ctrl.ready:
            qn = ctrl.explore(q)
            if qn is None:
                qn = q.copy()
            row["evt"] = "aim_explore"
        else:
            qn = ctrl.command(
                q, e, self.aim_target,
                gain=self.cfg["servo_gain"], max_dq=self.cfg["servo_max_dq"],
                tol=self.cfg["servo_tol"],
            )
            if qn is None:
                qn = q.copy()
            row["evt"] = "aim_step"
        self.aim_budget -= 1
        row["aim_err"] = round(err, 3)
        out = np.tile(st, (act.shape[0], 1))
        out[:, arm_slice] = qn
        out[:, LEFT_GRIPPER_IDX] = 1.0
        out[:, RIGHT_GRIPPER_IDX] = 1.0
        return out

    def _close_step(self, st, act, el, er, row):
        """Bounded harness-driven finger close at the aim spot. int301-a/b showed
        the VLA reliably fails to close even with fingers on the toy ("gave up"
        mode), so the harness finishes the mechanical last inch itself and hands
        back to the VLA for transport on confirm."""
        side = self.aim_arm
        gi = LEFT_GRIPPER_IDX if side == "l" else RIGHT_GRIPPER_IDX
        w = float(st[gi])
        self.close_hist.append(w)
        row["close_w"] = round(w, 3)
        # closed on nothing -> end immediately, and descend the hover height so
        # the NEXT aim reaches closer to the toy (int301-c: arrived at the toy
        # spot but closed 12cm above it -> every close grabbed air)
        if w < EMPTY_CLOSE_T:
            # v7: empty close usually means perception put the target ABOVE the
            # toy (int301-f aim#2: perceived +16cm too high, servo arrived
            # exactly there, fingers closed on air). Instead of walking away,
            # descend the aim point and re-approach within the SAME aim -- the
            # grasp sensor itself (gripper width) tells us when we hit paydirt.
            if self.close_retries < int(self.cfg.get("close_retries", 0)):
                self.close_retries += 1
                drop = float(self.cfg.get("descend_step", 0.06))
                self.aim_target = np.asarray(self.aim_target, dtype=float).copy()
                self.aim_target[2] -= drop
                self.phase = "aim"
                self.aim_budget = int(self.cfg.get("reapproach_budget", 15))
                self.aim_best_err = None
                self.aim_err_hist = []
                row["evt"] = "close_retry_descend"
                row["aim_target"] = [round(float(x), 3) for x in self.aim_target]
                logger.info(f"R9H: empty close -> descend retry #{self.close_retries} "
                            f"(target z -{drop} -> {self.aim_target[2]:.2f})")
                return None
            self.phase = "yield"
            self.yield_left = int(self.cfg["yield_fail_steps"])
            row["evt"] = "close_empty"
            if bool(self.cfg.get("hover_adapt", True)):
                new = max(self.hover_z - float(self.cfg["hover_adapt_step"]),
                          float(self.cfg["hover_min"]))
                if new != self.hover_z:
                    row["hover_adapt"] = [round(self.hover_z, 3), round(new, 3)]
                    self.hover_z = new
                    logger.info(f"R9H: empty close -> hover_z {row['hover_adapt'][0]} "
                                f"-> {row['hover_adapt'][1]} for next aim")
            logger.info(f"R9H: close on nothing (w={w:.3f}, {self.close_retries} retries "
                        f"used) -> short yield")
            return None
        # width plateaued mid-travel well above the empty band while commanded
        # closed -> fingers blocked by a small object (tennis-ball case:
        # int301-b aim#4 showed w pinned ~0.42-0.52 with fingers on the ball)
        if (len(self.close_hist) >= 4
                and max(self.close_hist[-4:]) - min(self.close_hist[-4:]) < 0.03
                and w < 0.85):
            self.grip[side]["state"] = "holding"
            self.holding = True
            self.aim_got_grasp = True
            self.phase = "yield"
            self.yield_left = int(self.cfg["yield_steps"])
            row["evt"] = "grasp_confirm_harness"
            logger.info(f"R9H: HARNESS GRASP (w plateau {w:.3f}) -> yield to VLA for transport")
            return None
        if self.close_left <= 0:
            self.phase = "yield"
            self.yield_left = int(self.cfg["yield_fail_steps"])
            row["evt"] = "close_fail"
            logger.info(f"R9H: close window out (w={w:.3f}) -> short yield")
            return None
        self.close_left -= 1
        row["evt"] = "close_step"
        out = np.tile(self.frozen_state, (act.shape[0], 1))
        out[:, gi] = -1.0          # command this gripper closed; freeze the rest
        return out


_HARNESS = None


def get_harness():
    """Env-gated singleton: B1K_R9_HARNESS=1 activates; anything else = pure passthrough."""
    global _HARNESS
    if _HARNESS is None:
        if os.environ.get("B1K_R9_HARNESS", "0") == "1":
            try:
                _HARNESS = R9Harness()
                logger.info("R9H: harness ACTIVE")
            except Exception:
                logger.exception("R9H: init failed; harness disabled")
                _HARNESS = None
        else:
            logger.info("R9H: env gate off; harness disabled")
    return _HARNESS
