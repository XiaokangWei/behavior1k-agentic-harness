import json
import numpy as np

rows = []
with open("/out/toys-probe3/telemetry.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))
snaps = [r for r in rows if r.get("evt") == "snap"]

# For each candidate: compare (ds/gd) against ray-length ratio sqrt(1+tx^2+ty^2)
# and against 1.0 (z-depth). Right hypothesis -> tight match on one of the two.
stats = {}
det = []
for s in snaps:
    for o in s.get("objs", []):
        gd = o.get("geom_d")
        if not gd or gd < 0.05:
            continue
        for c in o.get("uv_cands", []):
            fcx, u, v, ds = c
            if ds <= 0.01:
                continue
            tx = (u - 360.0) / float(fcx)
            ty = (v - 360.0) / 800.0
            ratio = float(np.sqrt(1 + tx * tx + ty * ty))
            key = float(fcx)
            stats.setdefault(key, {"raydiff": [], "zdiff": []})
            stats[key]["raydiff"].append(abs(ds / gd - ratio))
            stats[key]["zdiff"].append(abs(ds / gd - 1.0))
            if len(det) < 14 and key == 306.0:
                det.append((o["cat"], s["step"], fcx, u, v, round(ds, 2), gd,
                            round(ds / gd, 3), round(ratio, 3)))

for k in sorted(stats):
    st = stats[k]
    n = len(st["raydiff"])
    print(f"fx={k:5.0f} n={n:3d}  |ds/gd-rayratio|={np.mean(st['raydiff']):.3f}  |ds/gd-1|={np.mean(st['zdiff']):.3f}")

print()
print("details for fx=306 samples (cat, step, fx, u, v, ds, gd, ds/gd, ray_ratio):")
for d in det:
    print(" ", d)
