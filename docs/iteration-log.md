# Iteration Log — task 07 picking_up_toys, instance 301 (r9 checkpoint)

14 full evaluation runs (28k+ sim steps each, ~33 min wall time). Every version
change below was driven by forensic evidence from the run named in parentheses.
Harness direct contribution to score: **0 so far** — every q_score below was
produced by the VLA policy itself; the harness never completed a real grasp.

| run | score | harness | verdict of the post-run forensics |
|---|---|---|---|
| a | 0.333 | v1 | first intervention; aims stalled far from target |
| b | 0.167 | v1 | 4 aims burnt 1200+1500 steps each, killed the 4th attempt |
| c | 0.5 | v2 | stall watchdog + close phase added; last-cm miss (hover 12 cm) |
| d | 0.333 | v3 | pre_grasp_offset 0.12 → 0.03; 4 no_blob retries burned all perception credits in 60 steps |
| e | 0.167 | v4 | profile + hover adaptation added; 5 aims ALL hit furniture |
| f | 0.667 | v5 | VLA hot streak; quadruple filter being designed |
| g | 0.333 | v6 | shape/z-band filters in; explore deadlock on dead joints found |
| h | 0.0 | v7 | pure VLA variance; exposed that targets sit 1.3–3.1 m out |
| i | 0.0 | v8.5 | key calibration run: bq = base velocity; toys never in reach |
| j | 0.5 | v9 | approach drives the base (0.87 m proven) but loses target (FoV) |
| k | 0.167 | v9.1 | 12× same in-reach blob eef-gated while VLA fumbled (found) |
| l | 0.167 | v9.2 | target "creep" 0.79 → 1.42 m across chunks; head was frozen |
| m | 0.5 | v10 | first approach_arrived ever; pan servo works; aim plateau 0.09 m |
| n | 0.167 | v10.1 | z-band candidate priority + servo_tol 0.10; final validation run, q=0.167 (VLA self-completed without a harness grasp) |

## Failure → fix chain

- **v1 → v2 (run b):** aims without convergence burned the entire action budget.
  Added a stall watchdog (window 15 chunks, abort if improvement < 0.035) and a
  dedicated close phase (harness commands the gripper itself).
- **v2 → v3 (run c):** 12 cm hover offset closed grippers around air.
  `pre_grasp_offset` z: 0.12 → 0.03.
- **v3 → v4 (run d):** mode_by_task keyed `str(3)` vs `int(3)` silently fell
  back to intervene during the *plates* run → per-task key normalization.
- **v4 → v5 (run d/e):** no-blob retries without cooldown burned the
  perception-credit limit. Added `perception_retry_cooldown` (600 steps).
- **v5 → v6 (run e):** 12/12 aims hit warm furniture (tennis-ball at 2.2 m from
  the aimed point). Added the quadruple perception filter: aspect ≤ 3.0,
  fill ≥ 0.30, s_max 0.55, z-band [0.35, 0.95], eef_min_dist 0.25.
- **v6 → v7 (run g):** empty closes at hover height → re-descend retry
  (close_retries 2, descend_step 0.06); reach check moved to horizontal plane,
  max 0.85 m.
- **v7 → v8 (run g):** joint exploration deadlocked on dead joints (15 explore
  calls, zero new columns). Force-ready after 5 stalled explores.
- **v8 → v9 (runs h/i):** the structural wall. Ground truth: the VLA parks
  1.25–3.1 m from every toy, arm reach 0.85 m, VLA drives the base only 2.5 %
  of chunks. No aim-quality tuning can fix "unreachable". Added the approach
  phase: drive the base toward far candidates (action = normalized velocity,
  gains calibrated from run i telemetry), re-perceive every chunk.
- **v9 → v9.1 (run j):** mid-drive perception loss killed both approaches in
  ≤4 chunks. gain 0.8 → 0.5, miss_limit 3 → 8, jump gate 0.4 m, candidate
  selection = dist + 0.5·|cand − prev_est|.
- **v9.1 → v9.2 (run k):** the same in-reach blob was eef-gated 12× while the
  VLA fumbled at it for 7000 steps (harness watched in cooldown limbo). Fumble
  rescue: 4 same-position (±0.2 m) eef-gated triggers force a rescue aim;
  empty rescue blacklists the spot.
- **v9.2 → v10 (run l):** target estimate crept 0.79 → 1.42 m across furniture
  blobs because the head was frozen while driving. URDF: `torso_joint4`
  (action dim 6, ±3.05 rad yaw) carries the zed camera → closed-loop head-pan
  servo during approach (±0.15 rad/chunk, camera pose read live). Estimate lock
  inside 0.7 m.
- **v10 → v10.1 (run m):** first `approach_arrived` ever (pan servo held the
  target 4 chunks, dbeta −0.69 → −0.19). But the aim target was a 645 px
  furniture rim at z=0.91 — in-reach candidates were ranked by pixel size.
  Strict toy z-band [0.45, 0.85] became the primary ranking key (in-reach and
  far). servo_tol 0.07 → 0.10: the aim converged 0.39 → 0.09 then plateaued
  above the gate and burned its budget instead of entering close.

## Known remaining bottleneck

Even inside arm reach, the joint-space servo can plateau (0.09–0.38 m residual)
when the arm starts from a bad configuration (run m aim#1: EEF wedged at z=0.13,
target z=0.56; run i: residual 0.17 at full extension). The learned-Jacobian
servo with random exploration cannot climb out. Planned but not shipped:
analytic IK for the Panda arms using the URDF joint parameters (already
extracted).

## Finalization (2026-10-14)

Iteration stopped after run n (the 14th run, v10.1). The v10.1 design — base
approach (drive + head-pan servo) + in-reach quad-filter + toy-z candidate
priority + fumble rescue + drive-stability gates — is the frozen harness. It is
committed to `configs/profile_task7.json` and `r9_harness.py`, and the whole
repo is published at https://github.com/XiaokangWei/behavior1k-agentic-harness.

Honest bottom line: **the harness has not yet produced a single score point on
its own** — every 0.0–0.667 came from VLA variance, and the bare-policy baseline
(mean 0.444) still beats the intervened mean (0.286). The harness is a complete,
forensically-justified intervention scaffold; turning it into a net-positive
controller is left as future work (the analytic-IK arm servo above is the
highest-leverage next step, since approach + perception are already solved).
