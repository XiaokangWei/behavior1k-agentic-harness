#!/usr/bin/env python3
"""Offline mock test for harness v11: ground-truth re-grounding (S2) + analytic
transport/release (S1), per HarnessVLA/HarnessPAI strategy doc.

Builds a synthetic obs that ALSO carries obs["r9_gt"] (robot-local toy/box poses),
the payload the sim-side gt_injector.py produces. Validates:
  G1. GT target selection skips in_box toys and picks the nearest in-reach toy
      (not the RGB-D blob) -- S2 scene re-grounding.
  G2. GT far candidate -> base approach drive (same as v10 path, GT-sourced).
  G3. grasp confirm -> analytic transport: carry to box (base=0, gripper CLOSED)
      then release (gripper OPEN) at the box-approach pose -> yield. S1.
No sim / no GPU. Run: python3 mock_test_v11.py
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, ".")
import r9_harness

H = W = 720
FX, FY, CX, CY = 306.0, 800.0, 360.0, 360.0


def make_obs(r9_gt=None, eef_l=(0.40, 0.30, 0.65), eef_r=(0.70, -0.20, 0.60)):
    proprio = np.zeros(61)
    proprio[17:20] = eef_l
    proprio[42:45] = eef_r
    obs = {"robot_r1::proprio": proprio}
    if r9_gt is not None:
        obs["r9_gt"] = r9_gt
    return obs


def make_state(base_vel=(0.0, 0.0, 0.0), gripper_l=1.0):
    st = np.zeros(23)
    st[0:3] = base_vel
    st[3:7] = 0.0
    st[7:14] = 0.1
    st[14] = gripper_l
    st[15:22] = -0.1
    st[22] = 1.0
    return st


def make_actions():
    act = np.zeros((26, 23))
    act[:, 14] = 1.0
    act[:, 22] = 1.0
    return act


def new_harness(use_gt=True, transport=True):
    h = r9_harness.R9Harness.__new__(r9_harness.R9Harness)
    h.config_path = "/tmp/nonexistent_r9h_config.json"
    h._user_cfg = {
        "mode": "intervene", "tasks": [7], "log_path": "/tmp/r9h_mock_v11.jsonl",
        "dwell_trigger": 3000, "stage_min": 2, "stage_max": 13,
        "mode_by_task": {"7": "intervene"},
        "use_gt_poses": use_gt,
        "analytic_transport_after_grasp": transport,
        "gt_box_offset": [0.0, 0.0, 0.18],
        "gt_transport_tol": 0.12,
        "hold_min_width": 0.06,
    }
    h._cfg_mtime = None
    h.task_id = None
    h._reset_episode(7)
    return h


def prime_trigger(h, empty=5, no_att=2000, dwell=5000):
    h.empty_since_progress = empty
    h.no_attempt_steps = no_att
    h.dwell = dwell
    h.stage = 5
    h.best_stage = 5          # avoid __call__ resetting dwell to 0 on stage advance
    h.n_interventions = 0
    h.perception_fails = 0
    h.far_cooldown_left = 0
    h.holding = False


def call(h, step, r9_gt, base_vel=(0.0, 0.0, 0.0), eef_l=(0.40, 0.30, 0.65), stage=5):
    obs = make_obs(r9_gt, eef_l=eef_l)
    sentinel = make_actions()
    out = h(7, stage, make_state(base_vel), sentinel, obs=obs, step_count=step)
    return out, (out is sentinel)


def rows():
    return [json.loads(l) for l in open("/tmp/r9h_mock_v11.jsonl")]


def test_g1_gt_target_selection():
    if os.path.exists("/tmp/r9h_mock_v11.jsonl"):
        os.remove("/tmp/r9h_mock_v11.jsonl")
    h = new_harness()
    gt = {
        "toys": [
            {"name": "toy_inbox", "lpos": [0.50, 0.0, 0.55], "in_box": True},
            {"name": "toy_far",   "lpos": [1.50, 0.0, 0.60], "in_box": False},
            {"name": "toy_near",  "lpos": [0.30, 0.0, 0.60], "in_box": False},
        ],
        "box": {"lpos": [0.60, 0.0, 0.50]},
    }
    prime_trigger(h)
    out, passthrough = call(h, 20000, gt)
    assert not passthrough, "G1: harness did not intervene"
    assert h.phase in ("aim", "close"), f"G1: expected aim/close, got {h.phase}"
    tgt = np.asarray(h.aim_target, dtype=float)
    assert abs(tgt[0] - 0.30) < 0.05, f"G1: picked wrong toy (got x={tgt[0]:.2f}, expected ~0.30; in_box toy must be skipped)"
    assert h._gt_box_lpos is not None, "G1: GT box pose not captured"
    print(f"G1 ok: GT picked nearest in-reach toy (x={tgt[0]:.2f}); in_box toy skipped; box captured={np.round(h._gt_box_lpos,2).tolist()}")


def test_g2_gt_approach_far():
    if os.path.exists("/tmp/r9h_mock_v11.jsonl"):
        os.remove("/tmp/r9h_mock_v11.jsonl")
    h = new_harness()
    gt = {
        "toys": [
            {"name": "toy_far", "lpos": [1.50, 0.0, 0.60], "in_box": False},
        ],
        "box": {"lpos": [0.60, 0.0, 0.50]},
    }
    prime_trigger(h)
    out, passthrough = call(h, 20000, gt)
    assert not passthrough, "G2: harness did not intervene"
    assert h.phase == "approach", f"G2: expected approach, got {h.phase}"
    assert out is not None and abs(out[0, 0]) > 1e-3, f"G2: base should drive, got {out[0,:3]}"
    print(f"G2 ok: GT far toy -> approach_start, base drive out[:,0]={out[0,0]:.3f}")


def test_g3_transport_after_grasp():
    if os.path.exists("/tmp/r9h_mock_v11.jsonl"):
        os.remove("/tmp/r9h_mock_v11.jsonl")
    h = new_harness(transport=True)
    h._gt_box_lpos = np.array([0.60, 0.0, 0.50], dtype=float)
    h.aim_arm = "l"
    h.phase = "close"
    h.aim_got_grasp = True
    # make the left-arm Jacobian "ready" so the transport servo runs
    for _ in range(7):
        h.ctrl["l"].known[:] = True
    st = make_state(gripper_l=0.20)
    act = make_actions()
    # close_hist plateau at 0.20 (>= hold_min_width, < 0.85) -> grasp confirm
    h.close_hist = [0.20, 0.20, 0.20, 0.20]
    row = {}
    out = h._close_step(st, act, np.array([0.40, 0.30, 0.60]), None, row)
    assert h.phase == "transport", f"G3: expected transport, got {h.phase} (row={row})"
    tt = np.asarray(h.transport_target, dtype=float)
    assert abs(tt[0] - 0.60) < 1e-6 and abs(tt[2] - 0.68) < 1e-6, f"G3: transport target wrong: {tt}"
    print(f"G3a ok: grasp confirm -> transport to box-approach {np.round(tt,3).tolist()}")
    # transport step far from target: base zeroed, gripper CLOSED (-1), arm moves
    far_eef = tt + np.array([0.5, 0.0, 0.0])
    out = h._transport_step(st, act, far_eef, None, {})
    assert out is not None, "G3: transport_step must return actions"
    assert abs(out[0, 0]) < 1e-9 and abs(out[0, 1]) < 1e-9 and abs(out[0, 2]) < 1e-9, \
        f"G3: base must be zeroed during carry, got {out[0,:3]}"
    assert out[0, 14] < -0.9, f"G3: gripper must stay CLOSED while carrying, got {out[0,14]}"
    # transport step AT target: release (gripper OPEN) then yield
    h.transport_budget = 5
    out = h._transport_step(st, act, tt.copy(), None, {})
    assert out is not None, "G3: release must return actions"
    assert out[0, 14] > 0.9, f"G3: must OPEN gripper at box, got {out[0,14]}"
    assert h.phase == "yield", f"G3: expected yield after release, got {h.phase}"
    print("G3b ok: at box -> gripper OPENED (release) + yield; base stayed zeroed during carry")


def test_g4_no_transport_without_gt():
    if os.path.exists("/tmp/r9h_mock_v11.jsonl"):
        os.remove("/tmp/r9h_mock_v11.jsonl")
    h = new_harness(transport=True)
    h._gt_box_lpos = None        # no GT box -> must NOT enter transport
    h.aim_arm = "l"
    h.phase = "close"
    h.aim_got_grasp = True
    for _ in range(7):
        h.ctrl["l"].known[:] = True
    st = make_state(gripper_l=0.20)
    act = make_actions()
    h.close_hist = [0.20, 0.20, 0.20, 0.20]
    out = h._close_step(st, act, np.array([0.40, 0.30, 0.60]), None, {})
    assert h.phase == "yield", f"G4: expected yield (no GT box), got {h.phase}"
    print("G4 ok: analytic_transport disabled when GT box absent -> yield (v10.1 behavior)")


if __name__ == "__main__":
    test_g1_gt_target_selection()
    test_g2_gt_approach_far()
    test_g3_transport_after_grasp()
    test_g4_no_transport_without_gt()
    print("\nALL V11 MOCK TESTS PASSED")
