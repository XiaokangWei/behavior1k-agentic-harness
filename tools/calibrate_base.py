#!/usr/bin/env python3
"""Calibrate the base action->motion mapping from a logged run.
Inputs: harness jsonl (bq=base state x/y/rz per re-plan, ba=VLA base action median)
        telemetry.jsonl (base world pose per snapshot)
Outputs:
  1. bq frame check: is st[0:3] relative-to-episode-start? vs telemetry world pose
  2. action gain: regression of per-chunk displacement vs commanded action
Usage: calibrate_base.py <harness.jsonl> <telemetry.jsonl>
"""
import json, math, sys
import numpy as np

hp, tp = sys.argv[1], sys.argv[2]
evs = [json.loads(l) for l in open(hp)]
evs = [e for e in evs if "bq" in e and "ba" in e and e.get("ba") is not None]
print(f"rows with bq/ba: {len(evs)}")

bq = np.array([e["bq"] for e in evs], dtype=float)
ba = np.array([e["ba"] for e in evs], dtype=float)
steps = np.array([e["step"] for e in evs], dtype=float)

print("\n== state ranges (bq) ==")
for i, nm in enumerate(["x", "y", "rz"]):
    print(f"  {nm}: {bq[:, i].min():.3f} .. {bq[:, i].max():.3f}  (span {bq[:, i].max()-bq[:, i].min():.3f})")
print("== action ranges (ba) ==")
for i, nm in enumerate(["ax", "ay", "arz"]):
    print(f"  {nm}: {ba[:, i].min():.3f} .. {ba[:, i].max():.3f}")

# frame check vs telemetry world pose
rows = [json.loads(l) for l in open(tp)]
tel = [(r.get("step", 0), r["base"][:2]) for r in rows[1:] if r.get("base")]
if tel:
    ts = np.array([t[0] for t in tel], dtype=float)
    wx = np.array([t[1][0] for t in tel], dtype=float)
    wy = np.array([t[1][1] for t in tel], dtype=float)
    # interpolate world pose at harness steps
    hx = np.interp(steps, ts, wx)
    hy = np.interp(steps, ts, wy)
    print("\n== frame check: bq vs (world - world@step0) ==")
    ref_x = hx - hx[0]
    ref_y = hy - hy[0]
    for nm, b, r in (("x", bq[:, 0], ref_x), ("y", bq[:, 1], ref_y)):
        span_b = b.max() - b.min()
        span_r = r.max() - r.min()
        c = np.corrcoef(b, r)[0, 1] if span_b > 1e-6 and span_r > 1e-6 else float("nan")
        print(f"  bq_{nm} vs world disp: corr={c:.3f}  span_bq={span_b:.3f} span_world={span_r:.3f}")
    # scale check if correlated
    span_bq0 = bq[:, 0].max() - bq[:, 0].min()
    if abs(np.corrcoef(bq[:, 0], ref_x)[0, 1]) > 0.8 and span_bq0 > 0.1:
        sc = span_r / span_bq
        print(f"  -> bq_x appears to be world displacement x {sc:.2f}")

# action -> displacement regression (per chunk, 20 sim steps)
print("\n== action gain regression (Δbq over next chunk vs ba this chunk) ==")
for i, nm in enumerate(["ax", "ay", "arz"]):
    a = ba[:-1, i]
    d = bq[1:, i] - bq[:-1, i]
    if a.std() < 1e-6:
        print(f"  {nm}: VLA never commands this dim (std~0) -- no data")
        continue
    A = np.vstack([a, np.ones_like(a)]).T
    coef, res, *_ = np.linalg.lstsq(A, d, rcond=None)
    pred = A @ coef
    r2 = 1 - (d - pred).var() / d.var() if d.var() > 1e-9 else float("nan")
    print(f"  {nm}: gain={coef[0]:.4f}/action-unit, bias={coef[1]:.4f}, R2={r2:.3f}")
    # nonlinearity check: gains for small vs large |action|
    m = np.abs(a) < 0.3
    if m.sum() > 10 and a[m].std() > 1e-4:
        A2 = np.vstack([a[m], np.ones_like(a[m])]).T
        c2 = np.linalg.lstsq(A2, d[m], rcond=None)[0][0]
        print(f"      gain for |a|<0.3: {c2:.4f}")

# saturation: displacement per chunk when action maxed
print("\n== per-chunk displacement stats (20 steps) ==")
d_all = bq[1:] - bq[:-1]
print(f"  |Δx|: p50={np.median(np.abs(d_all[:,0])):.4f} p95={np.percentile(np.abs(d_all[:,0]),95):.4f} max={np.abs(d_all[:,0]).max():.4f}")
print(f"  |Δy|: p50={np.median(np.abs(d_all[:,1])):.4f} p95={np.percentile(np.abs(d_all[:,1]),95):.4f} max={np.abs(d_all[:,1]).max():.4f}")
print(f"  |Δrz|: p50={np.median(np.abs(d_all[:,2])):.4f} p95={np.percentile(np.abs(d_all[:,2]),95):.4f} max={np.abs(d_all[:,2]).max():.4f}")
