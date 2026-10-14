#!/usr/bin/env python3
"""Ground-truth validation: were v6 aim targets real toys?
Compares harness aim targets (robot-local frame) against toy lpos
(robot-local) from telemetry at the same step.
Usage: validate_aims_gt.py <harness.jsonl> <telemetry.jsonl>
"""
import json, math, sys
from bisect import bisect_left

harness_p, tel_p = sys.argv[1], sys.argv[2]

evs = [json.loads(l) for l in open(harness_p)]
aims = []
cur = None
for e in evs:
    if e.get("aim_target") is not None and (cur is None or e["step"] != cur.get("step", -1)):
        # new aim group starts when aim_target appears after non-aim rows
        if cur is not None and e["step"] - cur["last"] > 60:
            cur = None
        if cur is None:
            cur = {"step": e["step"], "target": e["aim_target"], "arm": e.get("arm"),
                   "last": e["step"], "errs": []}
            aims.append(cur)
    if cur is not None and e.get("evt", "").startswith("aim"):
        cur["last"] = e["step"]
        if e.get("aim_err") is not None:
            cur["errs"].append(e["aim_err"])
        if e.get("evt") in ("aim_stall_abort", "aim_arrived", "aim_budget_out"):
            cur["outcome"] = e["evt"]

rows = [json.loads(l) for l in open(tel_p)]
snaps = [(r.get("step", 0), r) for r in rows[1:] if "objs" in r]
steps = [s for s, _ in snaps]
TOY_CATS = ("board_game", "tennis", "ball", "toy", "plush", "stuffed", "rattle",
            "puzzle", "block", "jigsaw", "car", "figure", "cube", "lego")

def nearest_toy(step, tgt):
    i = bisect_left(steps, step)
    i = min(i, len(steps) - 1)
    r = snaps[i][1]
    best = (1e9, None)
    for o in r["objs"]:
        if o["cat"] in ("agent", "floor") or o["cat"] not in TOY_CATS:
            continue
        lp = o["lpos"]
        d = math.dist(tgt, lp)
        if d < best[0]:
            best = (d, o)
    # also nearest non-toy for contrast
    bestnt = (1e9, None)
    for o in r["objs"]:
        if o["cat"] in ("agent", "floor"):
            continue
        lp = o["lpos"]
        d = math.dist(tgt, lp)
        if d < bestnt[0]:
            bestnt = (d, o)
    return best, bestnt

print(f"{len(aims)} aims found")
for a in aims:
    tgt = a["target"]
    (bd, bo), (nd, no) = nearest_toy(a["step"], tgt)
    horiz = math.hypot(tgt[0], tgt[1])
    errs = a["errs"]
    print(f"\naim step={a['step']} arm={a['arm']} target={[round(x,2) for x in tgt]} "
          f"horiz={horiz:.2f} z={tgt[2]:.2f}")
    print(f"  servo: dist0 -> min_err={min(errs):.3f} outcome={a.get('outcome')}")
    if bo:
        print(f"  NEAREST TOY:  {bo['name'][:28]:28s} lpos={[round(x,2) for x in bo['lpos']]} dist={bd:.2f}m")
    if no:
        print(f"  nearest any:  {no['name'][:28]:28s} lpos={[round(x,2) for x in no['lpos']]} dist={nd:.2f}m")
    print(f"  VERDICT: {'REAL TOY' if bd < 0.30 else 'NOT a toy (or toy moved)'}")
