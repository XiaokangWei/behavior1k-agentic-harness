#!/usr/bin/env python3
"""Post-run episode forensics for one harness jsonl + result json.
Usage: analyze_episode.py <r9_harness.jsonl> [result_json]
"""
import json, sys

jp = sys.argv[1]
evs = [json.loads(l) for l in open(jp)]
print("rows:", len(evs), "first step:", evs[0].get("step"), "last step:", evs[-1].get("step"))

# 1. stage trajectory
adv = [(e["step"], e.get("stage")) for e in evs if e.get("evt") == "stage_advance"]
print("\n== stage advances ==")
for s, st in adv:
    print(f"  step {s} -> stage {st}")
print("final stage:", evs[-1].get("stage"), "best:", evs[-1].get("best"))

# 2. gripper events (VLA grasp confirmations)
ge = [(e["step"], e["gevents"]) for e in evs if e.get("gevents")]
print("\n== gripper events ==")
for s, g in ge[:20]:
    print(f"  step {s}: {g}")

# 3. harness events timeline
print("\n== harness events ==")
for e in evs:
    evt = e.get("evt")
    if evt in ("aim_start", "aim_arrived", "aim_stall_abort", "aim_budget_out",
               "close_empty", "close_grab", "yield_end", "no_blob", "target_far",
               "blacklist", "hover_adapt", "perception_fail", "config"):
        keep = {k: e[k] for k in ("step", "stage", "evt", "aim_err", "hover_z",
                                  "n_int", "arm", "aim_target", "holding", "dwell") if k in e}
        print(" ", keep)

# 4. no_blob context: what was the robot doing around each no_blob
nb = [e for e in evs if e.get("evt") in ("no_blob", "perception_fail")]
if nb:
    print("\n== no_blob context ==")
    for e in nb:
        s = e["step"]
        # find nearest rows before/after for gripper/pose context
        ctx = [r for r in evs if abs(r["step"] - s) <= 200]
        lw = [r.get("lw") for r in ctx if r.get("lw") is not None]
        rw = [r.get("rw") for r in ctx if r.get("rw") is not None]
        hold = [r.get("holding") for r in ctx]
        print(f"  step {s} stage={e.get('stage')} dwell={e.get('dwell')} "
              f"lw~{min(lw) if lw else None}/{max(lw) if lw else None} "
              f"rw~{min(rw) if rw else None}/{max(rw) if rw else None} "
              f"holding~{max(hold) if hold else None}")

# 5. VLA stuck intervals (long dwell runs with no stage advance)
print("\n== stuck intervals (dwell>1500) ==")
runs = []
cur = None
for e in evs:
    d = e.get("dwell", 0)
    if d and d > 1500:
        if cur and e["step"] - cur[-1][0] <= 40:
            cur.append((e["step"], d))
        else:
            if cur and len(cur) > 2:
                runs.append(cur)
            cur = [(e["step"], d)]
if cur and len(cur) > 2:
    runs.append(cur)
for r in runs:
    print(f"  steps {r[0][0]}-{r[-1][0]} ({r[-1][0]-r[0][0]} steps) peak dwell {max(x[1] for x in r)} stage {evs[-1].get('stage')}")

# 6. result score
if len(sys.argv) > 2:
    d = json.load(open(sys.argv[2]))
    print("\n== result ==")
    print("  q_score:", d.get("q_score"), "success:", d.get("success"))
