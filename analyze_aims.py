#!/usr/bin/env python3
"""Group aims by aim_target-bearing explore events; show each aim separately."""
import json, math, sys

path = sys.argv[1]
evs = [json.loads(l) for l in open(path) if l.strip()]

aims = []
cur = None
for e in evs:
    evt = e.get("evt")
    tgt = e.get("aim_target")
    if evt == "aim_explore" and tgt is not None:
        if cur:
            aims.append(cur)
        cur = {"start": e["step"], "target": tgt, "arm": e.get("arm"), "explore_errs": [],
               "servo_errs": [], "end": None, "budget_out": None, "blacklist": None}
        continue
    if cur is None:
        continue
    if evt == "aim_explore":
        cur["explore_errs"].append(e.get("aim_err"))
        cur["end"] = e["step"]
    elif evt == "aim_step":
        cur["servo_errs"].append(e.get("aim_err"))
        cur["end"] = e["step"]
        cur["el"] = e.get("el")
        cur["er"] = e.get("er")
    elif evt == "aim_budget_out":
        cur["budget_out"] = e["step"]
    elif evt == "yield_end":
        cur["blacklist"] = e.get("blacklisted")
if cur:
    aims.append(cur)

for i, a in enumerate(aims):
    tgt = a["target"]
    horiz = math.hypot(tgt[0], tgt[1])
    print(f"aim#{i+1} steps {a['start']}-{a['end']} arm={a['arm']} target=[{tgt[0]:.2f},{tgt[1]:.2f},{tgt[2]:.2f}] horiz={horiz:.2f} z={tgt[2]:.2f}")
    ex = [x for x in a["explore_errs"] if x is not None]
    sv = [x for x in a["servo_errs"] if x is not None]
    if ex:
        print(f"  explore: n={len(ex)} err {ex[0]:.3f}->{ex[-1]:.3f}")
    if sv:
        stride = max(1, len(sv) // 14)
        print(f"  servo:   n={len(sv)} err0={sv[0]:.3f} min={min(sv):.3f} final={sv[-1]:.3f} budget_out={a['budget_out']} blacklist={a['blacklist']}")
        print("  trace: " + " ".join(f"{sv[j]:.3f}" for j in range(0, len(sv), stride)))
    else:
        print(f"  (no servo rows) budget_out={a['budget_out']} blacklist={a['blacklist']}")
    eef = a.get("er") if a["arm"] == "r" else a.get("el")
    if eef and tgt:
        d3 = math.sqrt(sum((eef[k] - tgt[k]) ** 2 for k in range(3)))
        dh = math.hypot(eef[0] - tgt[0], eef[1] - tgt[1])
        print(f"  eef_end=[{eef[0]:.2f},{eef[1]:.2f},{eef[2]:.2f}] dist3D={d3:.3f} distH={dh:.3f}")
    print()
