import json
import numpy as np

FX, FY, CX, CY = 583.0, 800.0, 360.0, 360.0
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
        if d.get("evt") == "snap":
            snaps.append(d)

for s in snaps[:3]:
    cam7 = np.array(s["head_rel_base"])
    R, t = q2r(cam7[3:7]), cam7[:3]
    print(f"=== step {s['step']} cam rel base pos: {np.round(t, 3).tolist()}")
    for o in s["objs"]:
        cat = o["cat"]
        if cat in TOY or cat == "toy_box":
            p = np.array(o["lpos"])
            pc = R.T @ (p - t)
            if pc[2] < -0.1:
                d = -pc[2]
                u = CX + FX * pc[0] / d
                v = CY - FY * pc[1] / d
                print(f"  {cat:15s} lpos={np.round(p, 2).tolist()} -> uv=({u:.0f},{v:.0f}) depth={d:.2f} in_view={0 < u < 720 and 0 < v < 720}")
            else:
                print(f"  {cat:15s} behind camera, pc_z={pc[2]:.2f}")
    break

print("--- harness blobs near step 1000 ---")
with open("/diag/r9logs/r9_harness.jsonl") as f:
    shown = 0
    for line in f:
        d = json.loads(line)
        if abs(d.get("step", 0) - 1000) <= 200 and d.get("blobs"):
            print(d["step"], d["blobs"][:3])
            shown += 1
            if shown >= 4:
                break

print("--- back-project harness blobs at step~1000 and match toys ---")
s = snaps[0]
cam7 = np.array(s["head_rel_base"])
R, t = q2r(cam7[3:7]), cam7[:3]
toys = [(o["cat"] + "_" + o["name"].split("_")[-1], np.array(o["lpos"])) for o in s["objs"] if o["cat"] in TOY]
with open("/diag/r9logs/r9_harness.jsonl") as f:
    done = 0
    for line in f:
        d = json.loads(line)
        if abs(d.get("step", 0) - 1000) <= 200 and d.get("blobs"):
            for b in d["blobs"]:
                dd, u, v = b["d"], b["u"], b["v"]
                p_cam = np.array([(u - CX) * dd / FX, (CY - v) * dd / FY, -dd])
                p_loc = R @ p_cam + t
                best = min(((np.linalg.norm(p_loc - tp), nm) for nm, tp in toys))
                print(f"  blob band={b['band']} px={b['px']} -> p_loc={np.round(p_loc,2).tolist()} nearest={best[1]} dist={best[0]:.2f}")
            done += 1
            if done >= 3:
                break
