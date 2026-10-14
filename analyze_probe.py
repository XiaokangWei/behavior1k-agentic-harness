import json
import numpy as np

TOY = {"tennis_ball", "jigsaw_puzzle", "board_game", "toy_box", "desk", "bed", "floors"}


def rgb2hsv(px):
    r, g, b = [x / 255.0 for x in px]
    mx = max(r, g, b); mn = min(r, g, b)
    d = mx - mn
    if d < 1e-6:
        h = 0.0
    elif mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return round(h * 30.0, 1), round(d / mx if mx > 0 else 0.0, 2), round(mx, 2)  # h in 0-180


with open("/out/toys-probe1/telemetry.jsonl") as f:
    rows = [json.loads(line) for line in f]

snaps = [r for r in rows if r.get("evt") == "snap"]
print("snapshots:", len(snaps), "steps:", [s["step"] for s in snaps])
print("head_intr_est of first snap:", snaps[0].get("head_intr_est"))

for s in snaps:
    print(f"\n=== step {s['step']} stage(with base {np.round(s.get('base', [0]*7)[:3], 2).tolist()}) ===")
    for o in s.get("objs", []):
        if o["cat"] not in TOY:
            continue
        uv = o.get("uv")
        patch = o.get("patch_rgb")
        hs = ""
        if patch:
            hs = f" hsv(h,s,v)={rgb2hsv(patch)}"
        if uv:
            inview = 0 < uv[0] < 720 and 0 < uv[1] < 720
            print(f"  {o['cat']:14s} uv=({uv[0]:5.0f},{uv[1]:5.0f}) d={uv[2]:.2f} in_view={inview} patch={patch}{hs}")
        else:
            print(f"  {o['cat']:14s} (no uv)")
