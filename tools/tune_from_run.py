#!/usr/bin/env python3
"""Cross-episode meta-tuner: read a finished run's harness jsonl (+ result json),
summarize what the VLA and the harness did, propose knob adjustments, and FREEZE
converged knobs into profile_task{N}.json (the per-task "固化" artifact).

This is the plan-level closed loop: VLA results (stage curve, grasp events,
intervention outcomes, q_score) reshape the harness plan for the next episode.

Usage:
  python3 tune_from_run.py <run_dir_or_jsonl> [--task N] [--apply-suggest]
      [--freeze-config /path/r9_harness_config.json]
      [--freeze "k=v,k=v"]     # explicit knobs to freeze (v is json)
  # freeze writes profile_task{N}.json = current effective knobs + provenance
"""
import json, os, sys, glob, math
from collections import Counter

def jload(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def analyze(jsonl_path, task_id):
    evs = jload(jsonl_path)
    out = {"n_rows": len(evs)}
    last = evs[-1] if evs else {}
    out["final_step"] = last.get("step")
    out["best_stage"] = max((e.get("best", 0) for e in evs), default=0)

    # ---- stage curve / VLA stuck intervals ----
    adv = [(e["step"], e["stage"]) for e in evs if e.get("evt") == "stage_advance"]
    stuck = []
    if adv:
        # dwell between advances > 3000 steps = stuck interval
        for i in range(1, len(adv)):
            gap = adv[i][0] - adv[i - 1][0]
            if gap > 3000:
                stuck.append({"after_stage": adv[i - 1][1], "steps": gap,
                              "from": adv[i - 1][0], "to": adv[i][0]})
    tail = 0
    if adv:
        tail = (out["final_step"] or 0) - adv[-1][0]
        if tail > 3000:
            stuck.append({"after_stage": adv[-1][1], "steps": tail,
                          "from": adv[-1][0], "to": out["final_step"]})
    out["stage_advances"] = adv
    out["vla_stuck_intervals"] = stuck

    # ---- intervention outcomes ----
    aims = []
    cur = None
    for e in evs:
        evt = e.get("evt", "")
        # an aim's FIRST row is the only one carrying aim_target (regardless of
        # whether it went straight to servo with a warm Jacobian or explored)
        if e.get("aim_target") is not None:
            if cur:
                aims.append(cur)
            cur = {"step": e["step"], "target": e.get("aim_target"),
                   "arm": e.get("arm"), "d0": e.get("dist0"), "outcome": None, "err": None}
        elif cur is not None:
            if e.get("aim_err") is not None:
                cur["min_err"] = min(cur.get("min_err", 9e9), e["aim_err"])
                cur["last_err"] = e["aim_err"]
            if evt == "aim_arrived":
                cur["outcome"] = "arrived"
            elif evt == "aim_stall_abort":
                cur["outcome"] = "stalled"
                cur["err"] = e.get("aim_err")
            elif evt == "aim_stall_close_try":
                cur["outcome"] = "stall_close_try"
            elif evt == "aim_budget_out":
                cur["outcome"] = "budget_out"
            elif evt == "close_empty":
                cur["outcome"] = "empty_close"
            elif evt == "close_fail":
                cur["outcome"] = "close_fail"
            elif evt == "grasp_confirm_harness":
                cur["outcome"] = "HARNESS_GRASP"
            elif evt == "yield_end":
                if e.get("blacklisted") and cur["outcome"] is None:
                    cur["outcome"] = "yield_no_grasp"
    if cur:
        aims.append(cur)
    out["aims"] = aims
    out["n_int"] = last.get("n_int", 0)
    out["harness_grasps"] = sum(1 for a in aims if a["outcome"] == "HARNESS_GRASP")
    out["empty_closes"] = sum(1 for a in aims if a["outcome"] == "empty_close")
    out["stalled"] = sum(1 for a in aims if a["outcome"] == "stalled")

    # ---- VLA gripper events ----
    ge = Counter()
    for e in evs:
        for g in (e.get("gevents") or []):
            ge[g.get("evt")] += 1
    out["vla_grip_events"] = dict(ge)

    # ---- gripper VLA closes (from lw/rw + pending) rough ----
    out["triggers"] = Counter(e.get("evt") for e in evs if e.get("evt") in
                              ("target_far_cooldown", "perception_fail")).most_common()
    return out

def suggest(a, cfg):
    """Heuristic proposals from this run's outcomes. Human-in-the-loop default:
    print only; --apply-suggest writes them as base-config overrides."""
    s = []
    if a["harness_grasps"] == 0 and a["empty_closes"] >= 1:
        z = cfg.get("pre_grasp_offset", [0, 0, 0.03])[2]
        s.append({"knob": "pre_grasp_offset", "to": [0.0, 0.0, round(max(z - 0.03, 0.0), 3)],
                  "why": f"{a['empty_closes']} empty closes, 0 harness grasps -> hover lower"})
    if a["stalled"] >= 2 and a["harness_grasps"] == 0:
        s.append({"knob": "reach_h_max", "to": 1.02,
                  "why": f"{a['stalled']} stalled aims -> targets at arm limit waste cycles"})
    nf = dict(a.get("triggers", {})).get("perception_fail", 0)
    if nf >= 4:
        s.append({"knob": "_review_bands", "to": None,
                  "why": f"{nf} perception no_blob fails -> HSV band review for this task"})
    stuck_steps = 0
    first_aim_step = min((a["step"] for a in a["aims"]), default=None)
    for x in a["vla_stuck_intervals"]:
        if first_aim_step is None or x["from"] < first_aim_step:
            end = min(x["to"], first_aim_step) if first_aim_step else x["to"]
            stuck_steps += max(0, end - x["from"])   # clip at first intervention
    if stuck_steps > 8000:
        s.append({"knob": "dwell_trigger", "to": 2400,
                  "why": f"VLA wasted {stuck_steps} steps stuck BEFORE first aim -> fire earlier"})
    return s

def freeze(task_id, config_path, knobs=None, note=""):
    """固化: persist converged knobs into profile_task{N}.json. If the profile
    already exists, update it in place (it owns those keys); otherwise seed
    from the live base config."""
    d = os.path.dirname(config_path)
    pp = os.path.join(d, f"profile_task{task_id}.json")
    src = {}
    if os.path.exists(pp):
        src = {k: v for k, v in json.load(open(pp)).items() if not k.startswith("_")}
    if not src:
        src = json.load(open(config_path))
    prof = {k: src[k] for k in knobs if k in src} if knobs else dict(src)
    prof["_provenance"] = note
    json.dump(prof, open(pp, "w"), indent=1)
    return pp, prof

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    path = args[0]
    if os.path.isdir(path):
        jl = os.path.join(path, "..", "..", "r9logs")  # not used; keep simple
    # accept: jsonl file OR run dir with telemetry; find jsonl + result
    if path.endswith(".jsonl"):
        jsonl = path
    else:
        cands = glob.glob(os.path.join(path, "..", "..", "**", "r9_harness.jsonl"), recursive=True)
        jsonl = path if path.endswith(".jsonl") else None
    # result json
    rj = None
    d = os.path.dirname(path)
    for c in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
        try:
            j = json.load(open(c))
            if "q_score" in j:
                rj = c
                break
        except Exception:
            pass
    task_id = 7
    for i, a in enumerate(args):
        if a == "--task":
            task_id = int(args[i + 1])
        if a == "--result":
            rj = args[i + 1]
    cfg_path = "/home/bh1k/wsp/robot/harness_exp/r9logs/r9_harness_config.json"
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    a = analyze(jsonl, task_id)
    print("=" * 60)
    print(f"run: {jsonl}")
    if rj:
        q = json.load(open(rj))
        print(f"q_score: {q.get('q_score')}")
    print(f"best_stage: {a['best_stage']}, interventions: {a['n_int']}, "
          f"harness grasps: {a['harness_grasps']}, empty closes: {a['empty_closes']}, "
          f"stalled aims: {a['stalled']}")
    print(f"stage advances: {a['stage_advances']}")
    for st in a["vla_stuck_intervals"]:
        print(f"  VLA stuck {st['steps']} steps after stage {st['after_stage']} "
              f"({st['from']}->{st['to']})")
    for g, n in a["vla_grip_events"].items():
        print(f"  VLA gripper event {g}: {n}")
    for aim in a["aims"]:
        print(f"  aim@{aim['step']}: target={aim['target']} arm={aim.get('arm')} "
              f"d0={aim.get('d0')} min_err={round(aim.get('min_err', 9), 3)} "
              f"-> {aim['outcome']} (err={aim['err']})")
    print("=" * 60)
    sug = suggest(a, cfg)
    for s in sug:
        print(f"SUGGEST: {s['knob']} -> {s['to']}  [{s['why']}]")
    if "--apply-suggest" in args:
        for s in sug:
            if s["knob"].startswith("_"):
                continue
            if s["knob"] == "pre_grasp_offset":
                cfg["pre_grasp_offset"] = s["to"]
            else:
                cfg[s["knob"]] = s["to"]
        json.dump(cfg, open(cfg_path, "w"), indent=1)
        print(f"applied suggestions to {cfg_path}")
    for i, arg in enumerate(args):
        if arg == "--freeze":
            knobs = None
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                knobs = [kv.split("=")[0] for kv in args[i + 1].split(",")]
            note = f"q={json.load(open(rj))['q_score']['final'] if rj else '?'} run={jsonl}"
            out, prof = freeze(task_id, cfg_path, knobs, note)
            print(f"FROZE profile -> {out}: {json.dumps(prof)}")

if __name__ == "__main__":
    main()
