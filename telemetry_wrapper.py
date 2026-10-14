"""Sim-side telemetry wrapper -- instrumented bare run (task-level harness plan, step 1).

PURE LOGGING: never modifies actions or observations. All privileged state is
consumed only for offline attribution, exactly like the earlier priv_diag path.

Usage (run via run_sim_r9.sh, /diag is first on PYTHONPATH):
  WRAPPER=telemetry_wrapper.TelemetryWrapper ./run_sim_r9.sh picking_up_toys 0 <run_name>

Writes B1K_TEL_PATH (default /out/telemetry.jsonl, i.e. $OUT/telemetry.jsonl on host):
  evt=setup : per-camera intrinsics (projectionMatrix/focalLength/apertures) + pose rel base
              + goal predicate list + object-scope categories
  evt=snap   : every B1K_TEL_EVERY steps (default 1000):
               - per goal-predicate satisfaction (authoritative per-toy "inside box" signal)
               - every object-scope entity: world pos + pos in robot-local frame + category
               - head-image patch color sample around each entity's projected pixel
               - head camera pose rel base (moves with trunk)
"""
import json
import os
import time

import numpy as np
import torch as th

from omnigibson.eval.wrappers.rgbd_full_res_wrapper import RGBDFullResWrapper
from omnigibson.eval.utils.eval_utils import get_robot_camera_names
from omnigibson.utils import transform_utils as T


def _lst(x):
    try:
        a = np.asarray(x, dtype=float)
        return a.reshape(-1).tolist() if a.size else None
    except Exception:
        return None


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


class TelemetryWrapper(RGBDFullResWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._path = os.environ.get("B1K_TEL_PATH", "/out/telemetry.jsonl")
        self._every = int(os.environ.get("B1K_TEL_EVERY", "1000"))
        self._n = 0
        self._head = None      # head sensor name
        self._head_key = None  # head rgb obs key
        self._roles = {}
        self._setup()

    # ------------------------------------------------------------- helpers
    def _log(self, row):
        row["wall"] = time.time()
        try:
            with open(self._path, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
        except Exception as e:  # never break the rollout
            print(f"[telemetry] write failed: {e}", flush=True)

    def _robot(self):
        return self.env.robots[0]

    def _cam_pose_rel_base(self, sensor):
        robot = self._robot()
        base_pose = robot.get_position_orientation()
        pos, orn = sensor.get_position_orientation()
        rel_pos, rel_orn = T.relative_pose_transform(pos, orn, *base_pose)
        return th.cat([rel_pos, rel_orn])

    # ------------------------------------------------------------- setup
    def _setup(self):
        row = {"evt": "setup"}
        try:
            robot = self._robot()
            eval_cfg = getattr(self.env, "_eval_robot_config", {})
            cam_names = get_robot_camera_names(robot.name, eval_cfg)
            cams = {}
            for role, cam_name in cam_names.items():
                sensor_name = cam_name.split("::")[1]
                if sensor_name not in robot.sensors:
                    continue
                sensor = robot.sensors[sensor_name]
                if not hasattr(sensor, "image_height"):
                    continue
                try:
                    cp = dict(sensor.camera_parameters or {})
                except Exception:
                    cp = {}
                params = {}
                for k, v in cp.items():
                    conv = _lst(v)
                    if conv is None:
                        conv = _f(v)
                    if conv is not None:
                        params[k] = conv
                cams[role] = {
                    "sensor": sensor_name,
                    "h": _f(sensor.image_height),
                    "w": _f(sensor.image_width),
                    "params": params,
                    "pose_rel_base": _lst(self._cam_pose_rel_base(sensor)),
                }
                self._roles[role] = sensor_name
                if role == "head":
                    self._head = sensor_name
                    self._head_key = f"{robot.name}::{sensor_name}::rgb"
            row["cameras"] = cams
            row["cam_role_order"] = [str(r) for r in cam_names.keys()]
        except Exception as e:
            row["err"] = repr(e)
        try:
            task = self.env.task
            preds = []
            for oi, option in enumerate(task.ground_goal_state_options):
                for pred in option:
                    preds.append({"o": oi, "pred": str(pred)})
            row["goal_options"] = preds
        except Exception as e:
            row["goal_err"] = repr(e)
        try:
            scope = []
            for name, ent in self.env.task.object_scope.items():
                try:
                    scope.append({"name": str(name), "cat": str(getattr(ent, "category", ""))})
                except Exception:
                    pass
            row["object_scope"] = scope
        except Exception as e:
            row["scope_err"] = repr(e)
        self._log(row)

    # ------------------------------------------------------------- snapshot
    def _snapshot(self, obs, final=False):
        row = {"evt": "snap", "step": self._n, "final": bool(final)}
        robot = self._robot()
        # --- goal predicates (authoritative per-toy signal, same call as TaskMetric) ---
        try:
            task = self.env.task
            preds = []
            for oi, option in enumerate(task.ground_goal_state_options):
                for pi, pred in enumerate(option):
                    try:
                        sat = bool(pred.evaluate(task._evaluate_predicate))
                    except Exception:
                        sat = None
                    preds.append({"o": oi, "p": pi, "sat": sat})
            row["preds"] = preds
        except Exception as e:
            row["pred_err"] = repr(e)

        # --- robot base pose (world) ---
        try:
            bp, bo = robot.get_position_orientation()
            row["base"] = _lst(th.cat([bp, bo]))
        except Exception:
            pass

        # --- head camera pose rel base + head rgb patch sampling per object ---
        head_sensor = None
        head_img = None
        depth_img = None
        fx = fy = cx = cy = None
        cam_rel = None
        try:
            if self._head is not None and self._head in robot.sensors:
                head_sensor = robot.sensors[self._head]
                cam_rel = _lst(self._cam_pose_rel_base(head_sensor))
                row["head_rel_base"] = cam_rel
                # raw (pre-flatten) obs is nested: obs[robot_name][sensor_name]["rgb"];
                # the flat "robot_r1::<sensor>::rgb" key only exists post-_preprocess_obs.
                head_img = None
                depth_img = None
                if self._head_key in obs:
                    head_img = np.asarray(obs[self._head_key])
                elif isinstance(obs, dict):
                    sub = obs.get(robot.name)
                    if isinstance(sub, dict) and self._head in sub:
                        mod = sub[self._head]
                        if isinstance(mod, dict):
                            if "rgb" in mod:
                                head_img = np.asarray(mod["rgb"])
                            if "depth_linear" in mod:
                                depth_img = np.asarray(mod["depth_linear"])
                if head_img is not None:
                    h, w = head_img.shape[0], head_img.shape[1]
                    params = dict(head_sensor.camera_parameters or {})
                    fl = _f(params.get("cameraFocalLength", params.get("focalLength")))
                    ap = _lst(params.get("cameraAperture")) or [None, None]
                    hap, vap = (ap[0] if len(ap) > 0 else None), (ap[1] if len(ap) > 1 else None)
                    if fl and hap:
                        fx = fl / hap * w
                    if fl and vap:
                        fy = fl / vap * h
                    cx, cy = w / 2.0, h / 2.0
                    row["head_intr_est"] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": w, "h": h}
        except Exception as e:
            row["head_err"] = repr(e)

        # --- every object-scope entity: world pos + local pos + patch sample ---
        objs = []
        try:
            base_pos, base_orn = robot.get_position_orientation()
            for name, ent in self.env.task.object_scope.items():
                try:
                    if not hasattr(ent, "get_position_orientation"):
                        continue
                    wpos, worn = ent.get_position_orientation()
                    lpos, lorn = T.relative_pose_transform(wpos, worn, base_pos, base_orn)
                    rec = {
                        "name": str(name),
                        "cat": str(getattr(ent, "category", "")),
                        "wpos": _lst(wpos),
                        "lpos": _lst(lpos),
                    }
                    # project into head image, sample a 16x16 patch
                    if head_img is not None and head_sensor is not None:
                        try:
                            cpos, corn = head_sensor.get_position_orientation()
                            pcam, _ = T.relative_pose_transform(wpos, worn, cpos, corn)
                            pcam = np.asarray(pcam, dtype=float)
                            if pcam[2] < -0.1:  # in front of camera (looking along -z)
                                d = -pcam[2]
                                # candidate-intrinsic calibration: sample depth at the
                                # projected pixel under several (fx, fy) hypotheses; the
                                # right hypothesis makes sampled depth == geometric distance
                                cands = []
                                for fcx, fcy in ((fx, fy), (360.0, 800.0), (583.0, 800.0),
                                                 (583.0, 583.0), (360.0, 360.0), (306.0, 306.0)):
                                    if fcx is None or fcy is None:
                                        continue
                                    u = cx + fcx * pcam[0] / d
                                    v = cy - fcy * pcam[1] / d
                                    iu, iv = int(round(u)), int(round(v))
                                    if 4 <= iu < head_img.shape[1] - 4 and 4 <= iv < head_img.shape[0] - 4:
                                        ds = float(np.asarray(depth_img[iv - 2:iv + 3, iu - 2:iu + 3]).mean())
                                        cands.append([round(float(fcx), 0), round(float(u), 1), round(float(v), 1), round(ds, 3)])
                                if cands:
                                    rec["uv_cands"] = cands
                                    rec["geom_d"] = round(float(d), 3)
                                u = cx + fx * pcam[0] / d
                                v = cy - fy * pcam[1] / d
                                rec["uv"] = [round(float(u), 1), round(float(v), 1), round(float(d), 3)]
                                iu, iv = int(round(u)), int(round(v))
                                if 8 <= iu < head_img.shape[1] - 8 and 8 <= iv < head_img.shape[0] - 8:
                                    patch = np.asarray(head_img[iv - 8:iv + 8, iu - 8:iu + 8, :3], dtype=float)
                                    if patch.max() <= 1.5:
                                        patch = patch * 255.0
                                    rec["patch_rgb"] = [round(float(x), 1) for x in patch.reshape(-1, 3).mean(axis=0)]
                        except Exception:
                            pass
                    objs.append(rec)
                except Exception:
                    continue
            row["objs"] = objs
        except Exception as e:
            row["objs_err"] = repr(e)

        self._log(row)

    # ------------------------------------------------------------- step hook
    def step(self, action, n_render_iterations=1):
        obs, reward, terminated, truncated, info = super().step(action, n_render_iterations=n_render_iterations)
        self._n += 1
        try:
            if self._n % self._every == 0:
                self._snapshot(obs)
        except Exception as e:  # never break the rollout
            print(f"[telemetry] snapshot failed: {e}", flush=True)
        if terminated or truncated:
            try:
                self._snapshot(obs, final=True)
            except Exception:
                pass
        return obs, reward, terminated, truncated, info
