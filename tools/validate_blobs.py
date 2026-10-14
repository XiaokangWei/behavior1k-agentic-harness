import json
import numpy as np

TOY = {"tennis_ball", "jigsaw_puzzle", "board_game"}

snaps = []
with open("/out/toys-r9-inst301b/telemetry.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("evt") == "snap":
            snaps.append(d)

rows = []
with open("/diag/r9logs/r9_harness.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))
pt = [r for r in rows if r.get("ptarget")]
print(f"snapshots: {len(snaps)}  harness rows: {len(rows)}  with ptarget: {len(pt)}")

# join each ptarget with nearest snapshot's toy positions (robot-local frame)
dists = []
toy_hits = {}
for r in pt:
    step = r["step"]
    if not snaps:
        break
    s = min(snaps, key=lambda x: abs(x["step"] - step))
    if abs(s["step"] - step) > 1500:
        continue
    toys = [(o["cat"] + "_" + o["name"].split("_")[-1], np.array(o["lpos"]))
            for o in s.get("objs", []) if o["cat"] in TOY]
    p = np.array(r["ptarget"])
    best = min(((float(np.linalg.norm(p - tp)), nm) for nm, tp in toys))
    dists.append(best[0])
    toy_hits.setdefault(best[1], []).append(round(best[0], 2))

dists = np.array(dists)
print(f"blob->toy distance: n={len(dists)} mean={dists.mean():.3f} median={np.median(dists):.3f}")
print(f"  <0.15m: {(dists < 0.15).sum()}   <0.30m: {(dists < 0.30).sum()}   <0.50m: {(dists < 0.50).sum()}")
for nm, ds in toy_hits.items():
    print(f"  nearest-toy {nm}: n={len(ds)} median={np.median(ds):.2f}")

# also show per-band stats: which HSV band's blobs land on toys
band_stats = {}
for r in pt:
    step = r["step"]
    s = min(snaps, key=lambda x: abs(x["step"] - step))
    if abs(s["step"] - step) > 1500:
        continue
    toys = [(o["cat"] + "_" + o["name"].split("_")[-1], np.array(o["lpos"]))
            for o in s.get("objs", []) if o["cat"] in TOY]
    for b in r.get("blobs", []):
        # recompute back-projection for each blob with calibrated intrinsics
        cam = None
        if not s.get("head_rel_base"):
            continue
        cam7 = np.array(s["head_rel_base"])
        q = np.array([cam7[6], cam7[3], cam7[4], cam7[5]])  # xyzw -> wxyz
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        t = cam7[:3]
        d = b["d"]
        tx = (b["u"] - 360.0) / 306.0
        ty = (360.0 - b["v"]) / 800.0
        norm = np.sqrt(1 + tx * tx + ty * ty)
        p_cam = np.array([tx * d / norm, ty * d / norm, -d / norm])
        p_loc = R @ p_cam + t
        best = min(((float(np.linalg.norm(p_loc - tp)), nm) for nm, tp in toys))
        band_stats.setdefault(tuple(b["band"]), []).append(best[0])

print("\nper-band blob->toy distance:")
for band, ds in band_stats.items():
    ds = np.array(ds)
    print(f"  band={list(band)}: n={len(ds)} median={np.median(ds):.2f} <0.3m fraction={(ds < 0.3).mean():.2f}")
