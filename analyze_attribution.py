#!/usr/bin/env python3
"""Attribution analysis for the instrumented bare run (task-level harness plan, step 2).

Inputs:
  telemetry.jsonl  (sim side)  : setup + per-1000-step snapshots (preds, toy lpos, base, head pose)
  r9_harness.jsonl (policy side): per-re-plan rows (stage, grippers, eef, blobs, events)
  result json                  : final q_score

Output: per-toy attribution table + stage dwell + grasp event stats + blob->toy HSV mapping.
"""
import json
import sys
import math
import numpy as np

TEL = sys.argv[1] if len(sys.argv) > 1 else "/home/bh1k/wsp/robot/behavior_1k/inference/docker/out/toys-r9-inst301/telemetry.jsonl"
HAR = sys.argv[2] if len(sys.argv) > 2 else "/home/bh1k/wsp/robot/harness_exp/r9logs/r9_harness.jsonl"

FX, FY, CX, CY = 583.0, 800.0, 360.0, 360.0  # cameraFocalLength=17 / aperture 20.995,15.29 * 720

TOY_CATS = {"tennis_ball", "jigsaw_puzzle", "board_game"}


def quat2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


tel = load(TEL)
snaps = [r for r in tel if r.get("evt") == "snap"]
setup = next((r for r in tel if r.get("evt") == "setup"), None)
har = load(HAR)
print(f"telemetry snapshots: {len(snaps)}, harness rows: {len(har)}")

# ---------------- 1. per-toy inside-box trajectory ----------------
toy_names = {}
for s in snaps:
    for o in s.get("objs", []):
        if o["cat"] in TOY_CATS:
            toy_names[o["name"]] = o["cat"]
box = None
for s in snaps:
    for o in s.get("objs", []):
        if o["cat"] == "toy_box":
            box = o
            break
    if box:
        break
print("\n=== toys ===")
for name, cat in toy_names.items():
    short = name.split(".")[0] + "_" + name.split("_")[-1]
    first_in = None
    lpos_last = None
    for s in snaps:
        found = next((o for o in s.get("objs", []) if o["name"] == name), None)
        if found:
            lpos_last = found["lpos"]
    # inside status from final snapshot preds is aggregated; use position-based check per snap
    inside_steps = []
    for s in snaps:
        found = next((o for o in s.get("objs", []) if o["name"] == name), None)
        bx = next((o for o in s.get("objs", []) if o["cat"] == "toy_box"), None)
        if found and bx:
            d = np.linalg.norm(np.array(found["lpos"]) - np.array(bx["lpos"]))
            if d < 0.45:
                inside_steps.append(s["step"])
    if inside_steps:
        first_in = inside_steps[0]
    print(f"{short:22s} cat={cat:14s} first_inside_step={first_in} final_lpos={[round(x,2) for x in (lpos_last or [0,0,0])]}")

# preds sat over time (authoritative)
print("\n=== preds sat over time (authoritative inside flags) ===")
for s in snaps:
    sat = [p["sat"] for p in s.get("preds", [])]
    if any(sat):
        print(f"step {s['step']:6d}: sat={sum(1 for x in sat if x)}/6  {sat}")

# ---------------- 2. blob -> toy mapping (perception validation) ----------------
print("\n=== blob->toy back-projection check ===")
blob_rows = [r for r in har if r.get("blobs")]
toy_traj = {}  # name -> list of (step, lpos)
for s in snaps:
    for o in s.get("objs", []):
        if o["cat"] in TOY_CATS:
            toy_traj.setdefault(o["name"], []).append((s["step"], o["lpos"]))
head_traj = [(s["step"], s.get("head_rel_base")) for s in snaps if s.get("head_rel_base")]

if blob_rows and head_traj:
    matches = {}
    tested = 0
    for r in blob_rows:
        step = r["step"]
        # find closest head pose
        hp = min(head_traj, key=lambda t: abs(t[0] - step))
        if abs(hp[0] - step) > 1500:
            continue
        cam7 = np.array(hp[1])
        R, t = quat2rot(cam7[3:7]), cam7[:3]
        for b in r["blobs"]:
            d, u, v = b["d"], b["u"], b["v"]
            p_cam = np.array([(u - CX) * d / FX, (CY - v) * d / FY, -d])
            p_loc = R @ p_cam + t
            # nearest toy at closest snapshot step
            best = None
            for name, traj in toy_traj.items():
                tp = min(traj, key=lambda x: abs(x[0] - step))
                if abs(tp[0] - step) > 1500:
                    continue
                dist = np.linalg.norm(np.array(tp[1]) - p_loc)
                if best is None or dist < best[1]:
                    best = (name, dist)
            if best and best[1] < 0.30:
                key = (best[0], tuple(b["band"]))
                matches.setdefault(key, []).append(best[1])
                tested += 1
    for (name, band), errs in sorted(matches.items()):
        print(f"toy={name.split('.')[0]}_{name.split('_')[-1]:16s} band={list(band)} n={len(errs)} mean_err={np.mean(errs):.3f}m")

# ---------------- 3. stage dwell + grasp events ----------------
print("\n=== stage trajectory (policy side) ===")
stage_changes = []
last = None
for r in har:
    if r.get("evt") == "stage_advance" or (last is not None and r["stage"] != last):
        stage_changes.append((r["step"], r["stage"]))
    last = r["stage"]
print("stage transitions:", stage_changes[:40])

print("\n=== gripper events ===")
evts = {}
for r in har:
    for e in r.get("gevents", []):
        evts.setdefault(e["evt"], []).append((r["step"], e))
for k, v in evts.items():
    print(f"{k}: n={len(v)}")
    for step, e in v[:12]:
        print(f"   step {step}: {e}")

# ---------------- 4. eef vs toys ----------------
print("\n=== eef time near each toy (from harness trace, every 5th call) ===")
trace = [r for r in har if r.get("el") and r.get("blobs") is not None]
sample = [r for r in har if r.get("el") and (r.get("blobs") or r.get("gevents"))]
# eef proximity using snapshots' toy lpos
if trace:
    close_counts = {n: 0 for n in toy_traj}
    total = 0
    for r in trace[::2]:
        step = r["step"]
        for name, traj in toy_traj.items():
            tp = min(traj, key=lambda x: abs(x[0] - step))
            if abs(tp[0] - step) > 1500:
                continue
            for eef in (r.get("el"), r.get("er")):
                if eef and np.linalg.norm(np.array(eef) - np.array(tp[1])) < 0.35:
                    close_counts[name] += 1
                    break
        total += 1
    for name, c in close_counts.items():
        print(f"{name.split('.')[0]}_{name.split('_')[-1]:16s}: near-eef in {c}/{total} sampled calls")

print("\n=== empty closes summary ===")
empty_rows = [r for r in har if r.get("empty_sp", 0) > 0]
if empty_rows:
    print("last row:", empty_rows[-1])
print("final harness row:", har[-1] if har else None)
