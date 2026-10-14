import json
import numpy as np

FXC = [583.0, 360.0]      # fl/hap*720  vs  90deg hfov guess
FYC = [800.0, 583.0, 360.0]
CX, CY = 360.0, 360.0
TOY = {"tennis_ball", "jigsaw_puzzle", "board_game"}


def q2r(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


snaps = []
with open("/out/toys-r9-inst301/telemetry.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("evt") == "snap" and d.get("head_rel_base"):
            snaps.append(d)

blobs = []  # (step, [(u,v,d,px)])
with open("/diag/r9logs/r9_harness.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("blobs"):
            blobs.append((d["step"], [(b["u"], b["v"], b["d"], b["px"]) for b in d["blobs"][:4]]))
blob_steps = [b[0] for b in blobs]

results = {}
for fx in FXC:
    for fy in FYC:
        for qorder in ("wxyz", "xyzw"):
            for vsign in (-1, 1):
                hits = 0
                tot = 0
                for s in snaps[1:]:
                    cam7 = np.array(s["head_rel_base"])
                    if qorder == "xyzw":
                        q = np.array([cam7[6], cam7[3], cam7[4], cam7[5]])
                    else:
                        q = cam7[3:7]
                    R, t = q2r(q), cam7[:3]
                    # nearest blob row
                    if not blob_steps:
                        continue
                    bi = min(range(len(blob_steps)), key=lambda i: abs(blob_steps[i] - s["step"]))
                    if abs(blob_steps[bi] - s["step"]) > 300:
                        continue
                    bstep, bb = blobs[bi]
                    for o in s["objs"]:
                        if o["cat"] not in TOY:
                            continue
                        p = np.array(o["lpos"])
                        pc = R.T @ (p - t)
                        if pc[2] >= -0.2:
                            continue
                        d = -pc[2]
                        u = CX + fx * pc[0] / d
                        v = CY + vsign * fy * pc[1] / d
                        tot += 1
                        if not (0 < u < 720 and 0 < v < 720):
                            continue
                        for bu, bv, bd, px in bb:
                            if abs(bu - u) < 60 and abs(bv - v) < 60 and abs(bd - d) < 0.35:
                                hits += 1
                                break
                key = (fx, fy, qorder, vsign)
                results[key] = (hits, tot)

for k, v in sorted(results.items(), key=lambda kv: -kv[1][0]):
    print(f"fx={k[0]:5.0f} fy={k[1]:5.0f} quat={k[2]:5s} vsign={k[3]:+d}  hits={v[0]:4d}/{v[1]}")
