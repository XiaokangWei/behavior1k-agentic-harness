import json
import numpy as np

TOY = {"tennis_ball", "jigsaw_puzzle", "board_game"}


def rgb2hsv(px):
    r, g, b = [x / 255.0 for x in px]
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if d < 1e-6:
        h = 0.0
    elif mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return round(h * 30.0, 1), round(d / mx if mx > 0 else 0.0, 2), round(mx, 2)


# per-toy patch colors from probe3 (depth-candidate run)
for run in ("toys-probe3",):
    rows = []
    with open(f"/out/{run}/telemetry.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    snaps = [r for r in rows if r.get("evt") == "snap"]
    print(f"=== {run}: per-object patch colors (in-view only) ===")
    per = {}
    for s in snaps:
        for o in s.get("objs", []):
            if o.get("patch_rgb") and o["cat"] in TOY | {"toy_box", "desk", "floors", "bed"}:
                per.setdefault(o["cat"], []).append(o["patch_rgb"])
    for cat, patches in per.items():
        arr = np.array(patches)
        hsvs = [rgb2hsv(p) for p in patches]
        print(f"{cat:15s} n={len(patches):3d} mean_rgb={np.round(arr.mean(axis=0), 0).tolist()} "
              f"hsv range h[{min(h[0] for h in hsvs):.0f},{max(h[0] for h in hsvs):.0f}] "
              f"s[{min(h[1] for h in hsvs):.2f},{max(h[1] for h in hsvs):.2f}] "
              f"v[{min(h[2] for h in hsvs):.2f},{max(h[2] for h in hsvs):.2f}]")

# per-band -> per-toy hit matrix from the live run
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
pt = [r for r in rows if r.get("ptarget") or r.get("blobs")]

mat = {}
for r in pt:
    step = r["step"]
    if not snaps:
        break
    s = min(snaps, key=lambda x: abs(x["step"] - step))
    if abs(s["step"] - step) > 1500 or not s.get("head_rel_base"):
        continue
    toys = [(o["cat"] + "_" + o["name"].split("_")[-1], np.array(o["lpos"]))
            for o in s.get("objs", []) if o["cat"] in TOY]
    cam7 = np.array(s["head_rel_base"])
    q = np.array([cam7[6], cam7[3], cam7[4], cam7[5]])
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    t = cam7[:3]
    for b in r.get("blobs", []):
        d = b["d"]
        tx = (b["u"] - 360.0) / 306.0
        ty = (360.0 - b["v"]) / 800.0
        norm = np.sqrt(1 + tx * tx + ty * ty)
        p_cam = np.array([tx * d / norm, ty * d / norm, -d / norm])
        p_loc = R @ p_cam + t
        best = min(((float(np.linalg.norm(p_loc - tp)), nm) for nm, tp in toys))
        mat.setdefault(tuple(b["band"]), {}).setdefault(best[1] if best[0] < 0.3 else "none", []).append(best[0])

print("\n=== band -> toy hit matrix (<0.3m counts) from live run ===")
for band, toys_d in sorted(mat.items()):
    total = sum(len(v) for v in toys_d.values())
    hits = {k: len(v) for k, v in toys_d.items() if k != "none"}
    med = np.median([x for v in toys_d.values() for x in v])
    print(f"band={list(band)}: total={total} median_d={med:.2f} hits={hits}")
