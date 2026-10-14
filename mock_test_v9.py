#!/usr/bin/env python3
"""Offline mock test for harness v9 (approach phase).

Builds a synthetic obs: head camera at [0,0,1.5] pitched down 40deg (optical
axis forward-down), one warm toy blob at image center at distance d.
Simulates:
  A. trigger -> approach -> base-drive commands while far
  B. toy crosses into dexterity range -> approach_arrived -> aim (base=0)
  C. aim stall -> close attempt (base=0)
  D. blocked base (no motion) -> approach_blocked -> yield
No sim / no GPU needed. Run: python3 mock_test_v9.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import r9_harness

H = W = 720
FX, FY, CX, CY = 306.0, 800.0, 360.0, 360.0
ALPHA = np.deg2rad(45.0)   # camera pitch down from horizontal
CAM_T = np.array([0.0, 0.0, 1.55])   # head camera height in robot frame

# camera pose: optical axis -z_cam = (cos a, 0, -sin a) (forward-down)
R = np.array([
    [0, -np.sin(ALPHA), -np.cos(ALPHA)],
    [1, 0, 0],
    [0, -np.cos(ALPHA), np.sin(ALPHA)],
])


def world_to_pixel(X, Z):
    """Fixed world toy at (X, 0, Z) in robot frame -> (u, v, d) in the image."""
    pw = np.array([X, 0.0, Z]) - CAM_T
    p_cam = R.T @ pw
    d = float(np.linalg.norm(p_cam))
    tx = -p_cam[0] / p_cam[2]
    ty = -p_cam[1] / p_cam[2]
    u = CX + tx * FX
    v = CY - ty * FY          # flip_v: ty = (cy - v)/fy
    return u, v, d


def rot_to_quat_xyzw(Rm):
    """rotation matrix -> quat [x,y,z,w] (xyzw order)."""
    tr = Rm[0, 0] + Rm[1, 1] + Rm[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (Rm[2, 1] - Rm[1, 2]) / S
        y = (Rm[0, 2] - Rm[2, 0]) / S
        z = (Rm[1, 0] - Rm[0, 1]) / S
    else:
        i = int(np.argmax(np.diag(Rm)))
        if i == 0:
            S = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
            w = (Rm[2, 1] - Rm[1, 2]) / S
            x = 0.25 * S
            y = (Rm[0, 1] + Rm[1, 0]) / S
            z = (Rm[0, 2] + Rm[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
            w = (Rm[0, 2] - Rm[2, 0]) / S
            x = (Rm[0, 1] + Rm[1, 0]) / S
            y = 0.25 * S
            z = (Rm[1, 2] + Rm[2, 1]) / S
        else:
            S = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
            w = (Rm[1, 0] - Rm[0, 1]) / S
            x = (Rm[0, 2] + Rm[2, 0]) / S
            y = (Rm[1, 2] + Rm[2, 1]) / S
            z = 0.25 * S
    return np.array([x, y, z, w])


QUAT_XYZW = rot_to_quat_xyzw(R)
assert np.allclose(r9_harness._quat2rot(QUAT_XYZW[[3, 0, 1, 2]]), R, atol=1e-6), "quat roundtrip"

TOY_Z = 0.65          # toy height (table toy)
TOY_RGB = np.array([200.0, 160.0, 110.0])   # h=33.3deg s=0.45 v=0.78 -> band [25,40]


def make_obs(X, Z=TOY_Z, base_vel=(0.0, 0.0, 0.0),
             eef_l=(0.40, 0.30, 0.65), eef_r=(0.70, -0.20, 0.60)):
    rgb = np.zeros((H, W, 3), dtype=float)
    rgb[..., :] = [30.0, 30.0, 40.0]        # dark neutral background
    depth = np.full((H, W), 5.0, dtype=float)
    if X is not None:                        # X=None -> blank scene (no blob)
        u, v, d = world_to_pixel(X, Z)
        ui, vi = int(round(u)), int(round(v))
        assert 10 <= ui < W - 10 and 10 <= vi < H - 10, f"toy out of FoV: u={ui} v={vi}"
        rgb[vi - 10:vi + 10, ui - 10:ui + 10, :] = TOY_RGB
        depth[vi - 10:vi + 10, ui - 10:ui + 10] = d
    proprio = np.zeros(61)
    proprio[17:20] = eef_l
    proprio[42:45] = eef_r
    crp = np.zeros(21)
    crp[14:21] = [CAM_T[0], CAM_T[1], CAM_T[2], *QUAT_XYZW]
    return {
        "robot_r1::zed_head::rgb": rgb,
        "robot_r1::zed_head::depth_linear": depth,
        "robot_r1::proprio": proprio,
        "robot_r1::cam_rel_poses": crp,
    }


def make_state(base_vel=(0.0, 0.0, 0.0)):
    st = np.zeros(23)
    st[0:3] = base_vel
    st[3:7] = 0.0            # trunk/head
    st[7:14] = 0.1           # left arm joints
    st[14] = 1.0             # left gripper open
    st[15:22] = -0.1         # right arm
    st[22] = 1.0             # right gripper open
    return st


def make_actions():
    act = np.zeros((26, 23))
    act[:, 14] = 1.0
    act[:, 22] = 1.0
    return act


def new_harness():
    h = r9_harness.R9Harness.__new__(r9_harness.R9Harness)
    h.config_path = "/tmp/nonexistent_r9h_config.json"
    h._user_cfg = {
        "mode": "intervene", "tasks": [7], "log_path": "/tmp/r9h_mock_test.jsonl",
        "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
        "cam_pose_idx": 2, "quat_order": "xyzw", "depth_mode": "ray", "flip_v": True,
        "dwell_trigger": 3000, "stage_min": 2, "stage_max": 13,
        "mode_by_task": {"7": "intervene"},
    }
    h._cfg_mtime = None
    h._profile_dir = None
    h.task_id = None
    h._reset_episode(7)
    return h


def call(h, step, X, Z=TOY_Z, base_vel=(0.0, 0.0, 0.0), stage=5, eef_l=(0.40, 0.30, 0.65)):
    obs = make_obs(X, Z, base_vel, eef_l=eef_l)
    sentinel = make_actions()
    out = h(7, stage, make_state(base_vel), sentinel, obs=obs, step_count=step)
    return out, (out is sentinel)   # (actions, is_passthrough)


def rows():
    return [__import__("json").loads(l) for l in open("/tmp/r9h_mock_test.jsonl")]


def test_a_approach_triggers_and_drives():
    import os
    if os.path.exists("/tmp/r9h_mock_test.jsonl"):
        os.remove("/tmp/r9h_mock_test.jsonl")
    h = new_harness()
    X = 1.2   # fixed world toy at reach_h=1.2 -> FAR (beyond 0.85 arm reach)
    out = None
    n_pre = 0
    for k in range(400):
        step = k * 20
        out, passthrough = call(h, step, X)
        if not passthrough:
            break
        n_pre += 1
    assert out is not None, "A: harness never intervened (trigger did not fire)"
    rs = rows()
    ev = [r for r in rs if r.get("evt") == "approach_start"]
    assert ev, "A: expected approach_start event"
    print(f"A1 ok: trigger->approach after {n_pre} passthrough calls (dwell {ev[0]['dwell']})")
    # approach chunks must command base motion toward +x (target at +X)
    out2, _ = call(h, (n_pre + 1) * 20, X, base_vel=(0.3, 0.0, 0.0))
    assert out2 is not None and abs(out2[0, 0] - 0.8) < 0.05 and abs(out2[0, 1]) < 0.05, \
        f"A2: drive command wrong: {out2[0,:3]}"
    assert abs(out2[0, 2]) < 1e-9, "A2: yaw must be 0"
    assert out2[0, 14] > 0.9 and out2[0, 22] > 0.9, "A2: grippers must stay open"
    assert out2[0, 7] == 0.1, "A2: arms frozen at current state"
    print("A2 ok: drive cmd out[:,0:3]=", np.round(out2[0, :3], 3).tolist(),
          "grippers open, arms frozen")
    return h, n_pre


def test_b_arrival_to_aim(h, n_pre):
    # shrink X each chunk (base closing in at ~0.25m/chunk = 0.58 m/s * 0.67s * 0.8)
    k = n_pre + 1
    X = 1.2
    arrived = False
    while X > 0.55:
        X = max(0.55, X - 0.25)
        out, _ = call(h, k * 20, X, base_vel=(0.25, 0.0, 0.0))
        k += 1
        r = rows()[-1]
        if r.get("evt") == "approach_arrived" or (
                r.get("evt") == "aim_start" and r.get("ptarget")):
            arrived = True
        if h.phase == "aim":
            assert out is not None, "B: aim_step must return actions"
            assert abs(out[0, 0]) < 1e-9 and abs(out[0, 1]) < 1e-9 and abs(out[0, 2]) < 1e-9, \
                f"B: aim must zero base cmd, got {out[0,:3]}"
            break
    assert arrived, "B: approach_arrived never fired"
    assert h.phase == "aim", f"B: expected aim phase, got {h.phase}"
    print(f"B ok: approach_arrived -> aim (base cmd zeroed during servo); "
          f"aim_target={np.round(h.aim_target,3).tolist()}")


def test_c_close_zeroes_base(h):
    # force the close phase by setting servo_tol huge: next aim_step = arrived
    h._user_cfg["servo_tol"] = 99.0
    h.phase = "aim"
    h.aim_budget = 10
    h.aim_target = np.array([0.5, 0.0, 0.5])
    out, _ = call(h, 99000, 0.7)
    assert h.phase == "close", f"C: expected close phase, got {h.phase}"
    out, _ = call(h, 99020, 0.7)
    assert out is not None, "C: close_step must return actions"
    assert abs(out[0, 0]) < 1e-9 and abs(out[0, 1]) < 1e-9 and abs(out[0, 2]) < 1e-9, \
        f"C: close must zero base cmd, got {out[0,:3]}"
    gi = out[0, 14] if h.aim_arm == "l" else out[0, 22]
    assert gi < -0.9, "C: close must command gripper -1"
    print("C ok: close phase commands gripper closed, base cmd zeroed")


def test_d_blocked_aborts():
    import os
    if os.path.exists("/tmp/r9h_mock_test.jsonl"):
        os.remove("/tmp/r9h_mock_test.jsonl")
    h = new_harness()
    X = 1.2
    out = None
    n_pre = 0
    for k in range(400):
        out, passthrough = call(h, k * 20, X, base_vel=(0.0, 0.0, 0.0))  # base never moves
        if not passthrough:
            break
    assert out is not None, "D: trigger failed"
    # keep calling with zero velocity -> blocked after 3 chunks
    blocked = False
    for j in range(6):
        call(h, (400 + j) * 20, X, base_vel=(0.0, 0.0, 0.0))
        r = rows()[-1]
        if r.get("evt") == "approach_blocked":
            blocked = True
            break
    assert blocked, "D: approach_blocked never fired"
    assert h.phase == "yield", f"D: expected yield after blocked, got {h.phase}"
    # during yield: passthrough
    out, passthrough = call(h, 99000 * 20, X)
    assert passthrough, "D: yield must passthrough"
    print("D ok: blocked base -> approach_blocked -> yield passthrough "
          f"(far_cooldown_left={h.far_cooldown_left})")


def test_e_fumble_rescue():
    """v9.2: in-reach blob rejected ONLY by the anti-stuck EEF gate, seen at the
    same position fumble_escalate times -> harness takes over with an aim."""
    import os
    if os.path.exists("/tmp/r9h_mock_test.jsonl"):
        os.remove("/tmp/r9h_mock_test.jsonl")
    h = new_harness()
    X = 0.45                      # rh=0.45 -> IN REACH
    eef_l = (0.40, 0.0, 0.65)     # 0.05m from the blob -> eef-rejected
    rescued = False
    fumbles = []
    out = None
    for k in range(500):
        out, passthrough = call(h, k * 20, X, eef_l=eef_l)
        r = rows()[-1]
        if r.get("fumble"):
            fumbles.append(r["fumble"])
        if r.get("evt") == "fumble_rescue":
            rescued = True
            break
        if not passthrough:
            break                  # some other intervention fired -- fail below
    assert rescued, f"E: fumble_rescue never fired (fumbles seen: {fumbles}, phase={h.phase})"
    assert h.phase in ("aim", "close"), f"E: expected aim/close phase, got {h.phase}"
    assert out is not None, "E: rescue must return actions"
    if h.phase == "aim":
        assert abs(out[0, 0]) < 1e-9 and abs(out[0, 1]) < 1e-9, "E: aim must zero base cmd"
    tgt = np.asarray(h.aim_target, dtype=float)
    assert abs(tgt[0] - 0.45) < 0.06 and abs(tgt[1]) < 0.06, \
        f"E: rescue aimed at wrong target: {tgt}"
    print(f"E ok: fumble rescue after {len(fumbles)} eef-gated triggers "
          f"-> aim at {np.round(tgt, 3).tolist()} (phase={h.phase})")


def test_f_pan_servo_direction():
    """v10: head-pan servo commands trunk t4 toward the target bearing."""
    import os
    if os.path.exists("/tmp/r9h_mock_test.jsonl"):
        os.remove("/tmp/r9h_mock_test.jsonl")
    h = new_harness()
    X = 1.2
    out = None
    for k in range(400):
        out, passthrough = call(h, k * 20, X)
        if not passthrough:
            break
    assert out is not None, "F: trigger failed"
    # target estimate off to the LEFT (+y): camera yaw is 0 in this mock, so
    # the pan servo must command a POSITIVE t4 increment (CCW, x->y).
    # X=None -> blank scene: perception sees no blob, so the manual estimate
    # survives (dead-reckon chunk) and the pan servo runs on it.
    h.approach_target = np.array([1.0, 0.6, 0.65])   # bearing ~ +0.54 rad
    h.approach_dist = 1.17
    h.phase = "approach"
    out, _ = call(h, (k + 1) * 20, None, base_vel=(0.2, 0.0, 0.0))
    assert out is not None, "F: drive chunk must return actions"
    r = rows()[-1]
    assert "pan" in r and "dbeta" in r, f"F: pan fields missing: {r}"
    assert r["dbeta"] > 0.3, f"F: expected positive bearing error, got {r['dbeta']}"
    assert out[0, 6] > 0.05, f"F: pan must turn positive (CCW), got {out[0, 6]}"
    assert abs(out[0, 3]) < 1e-9 and abs(out[0, 4]) < 1e-9 and abs(out[0, 5]) < 1e-9, \
        "F: trunk pitch joints must stay frozen at 0"
    # symmetric: target to the RIGHT (-y) -> negative pan
    h.approach_target = np.array([1.0, -0.6, 0.65])
    out, _ = call(h, (k + 2) * 20, None, base_vel=(0.2, 0.0, 0.0))
    assert out[0, 6] < -0.05, f"F: pan must turn negative (CW), got {out[0, 6]}"
    print(f"F ok: pan servo dbeta={r['dbeta']:.3f} -> t4 cmd {out[0, 6]:.3f} "
          f"(pitch frozen; symmetric for -y)")


if __name__ == "__main__":
    h, n_pre = test_a_approach_triggers_and_drives()
    test_b_arrival_to_aim(h, n_pre)
    test_c_close_zeroes_base(h)
    test_d_blocked_aborts()
    test_e_fumble_rescue()
    test_f_pan_servo_direction()
    print("\nALL MOCK TESTS PASSED")
