#!/usr/bin/env python3
"""Analyze one harness jsonl run: interventions timeline, aim quality, outcomes."""
import json, sys, math
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/home/bh1k/wsp/robot/harness_exp/r9logs/r9_harness.jsonl"
events = []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        events.append(json.loads(line))
    except Exception:
        pass

print(f"total lines: {len(events)}")

# 1. key events (non aim_step/yield_step)
print("\n===== KEY EVENTS =====")
for e in events:
    evt = e.get("evt", "")
    if evt in ("aim_step",):
        continue
    step = e.get("step", -1)
    if evt == "reset":
        print(f"[{step:>6}] RESET task={e.get('task')}")
    elif "aim" in str(evt).lower() or "blacklist" in str(evt).lower() or "budget" in str(evt).lower() or "config" in str(evt).lower() or "yield_end" in str(evt).lower() or evt == "target_far_cooldown":
        tgt = e.get("target")
        extra = ""
        if tgt:
            extra = f" target=[{tgt[0]:.2f},{tgt[1]:.2f},{tgt[2]:.2f}]"
        if "dist0" in e:
            extra += f" dist0={e['dist0']:.3f}"
        if "arm" in e:
            extra += f" arm={e['arm']}"
        if "trigger" in e:
            extra += f" trigger={e['trigger']}"
        if "aim_err" in e:
            extra += f" err={e['aim_err']:.3f}"
        print(f"[{step:>6}] {evt}{extra}")

# 2. group aim_steps into aims by aim index / phase transitions
print("\n===== AIM TRAJECTORIES =====")
aims = []  # list of dicts
cur = None
last_phase = None
for e in events:
    phase = e.get("phase")
    evt = e.get("evt", "")
    if evt == "aim_start":
        cur = {"start": e.get("step"), "errs": [], "target": e.get("target"), "arm": e.get("arm"),
               "dist0": e.get("dist0"), "rows": 0, "min_err": 9e9, "end": None, "end_evt": None,
               "trigger": e.get("trigger"), "eef_at_end": None}
        aims.append(cur)
    if evt == "aim_budget_out" or "blacklist" in evt or evt == "yield_end":
        if cur is not None and cur.get("end") is None:
            cur["end"] = e.get("step")
            cur["end_evt"] = evt
    if evt == "aim_step" and cur is not None:
        err = e.get("aim_err")
        if err is not None:
            cur["errs"].append(err)
            cur["min_err"] = min(cur["min_err"], err)
            cur["end"] = e.get("step")
        el, er = e.get("el"), e.get("er")
        if el and er:
            cur["eef_at_end"] = (el, er)

for i, a in enumerate(aims):
    errs = a["errs"]
    tgt = a["target"]
    if not errs:
        print(f"aim#{i+1}: no trajectory rows")
        continue
    n = len(errs)
    print(f"aim#{i+1}: steps {a['start']}->{a['end']} ({(a['end'] or 0)-(a['start'] or 0)} steps, {n} samples)")
    print(f"   target={tgt} arm={a.get('arm')} dist0={a.get('dist0')}")
    print(f"   err: start={errs[0]:.3f} min={min(errs):.3f} final={errs[-1]:.3f} end_evt={a.get('end_evt')}")
    # print err every ~10 samples
    samp = [f"{errs[j]:.3f}" for j in range(0, n, max(1, n // 12))]
    print(f"   err trace: {' '.join(samp)}")
    if a.get("eef_at_end"):
        el, er = a["eef_at_end"]
        if a.get("arm") == "r":
            eef = er
        else:
            eef = el
        if tgt:
            dist = math.sqrt(sum((eef[k] - tgt[k]) ** 2 for k in range(3)))
            print(f"   eef(end)=[{eef[0]:.2f},{eef[1]:.2f},{eef[2]:.2f}] -> 3D dist to target = {dist:.3f}")

# 3. dwell / stage progression summary
print("\n===== STAGE PROGRESSION =====")
stage_changes = []
last_stage = None
for e in events:
    s = e.get("stage")
    if s is not None and s != last_stage:
        stage_changes.append((e.get("step", -1), s))
        last_stage = s
print(" ".join(f"{s}@{st}" for st, s in stage_changes if st is not None))

best = max((e.get("best", 0) for e in events), default=0)
print(f"best stage seen: {best}")

# 4. gripper close attempts (proxy: lw/rw dropping below 0.2 from >=0.8)
print("\n===== GRIPPER CLOSE EVENTS =====")
prev_l, prev_r = 1.0, 1.0
closes = 0
for e in events:
    lw, rw = e.get("lw"), e.get("rw")
    if lw is None or rw is None:
        continue
    if (prev_l >= 0.8 and lw < 0.2) or (prev_r >= 0.8 and rw < 0.2):
        closes += 1
        print(f"[{e.get('step')}] close (lw {prev_l:.2f}->{lw:.2f}, rw {prev_r:.2f}->{rw:.2f})")
    prev_l, prev_r = lw, rw
print(f"total closes: {closes}")
