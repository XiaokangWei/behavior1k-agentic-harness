# B1K R9 Harness — VLA Intervention Harness for BEHAVIOR-1K

An agentic intervention harness that wraps a frozen VLA policy (r9 checkpoint,
fm2048_step68203) during BEHAVIOR-1K evaluation and takes over when the policy
gets stuck. Developed and iterated end-to-end on **task 07 `picking_up_toys`,
instance 301** (14 full simulation runs, each ~28k steps / ~33 min).

## What it does

The harness sits between the eval wrapper and the policy (`eval_b1k_wrapper.r9.py`
calls it once per action chunk, 20 sim steps). When the policy dwells in the
same subtask stage while failing to grasp (empty closes accumulate, or no grasp
attempts at all), the harness intervenes:

```
idle ──trigger──► perceive ──► [far?] ──► approach (drive base + pan head) ─┐
                   │                                                      │
                   │ [in reach]                                           ▼
                   ▼                                                    aim
aim (joint-space servo, learned Jacobian w/ explore-freeze) ──► close (gripper)
   │                                                             │
   │ stall / budget out                                          ▼
   ▼                                                       yield ──► idle
abort ──► blacklist target ──► yield ──► idle
```

Key mechanisms (each one was added because a specific failure was forensically
proven in a real run — see `docs/iteration-log.md`):

| Mechanism | Version | Fixes |
|---|---|---|
| Stall watchdog + close phase | v2 | budget burnt on non-converging aims |
| Adaptive hover offset (loop-level) | v3/v7 | pre-grasp hover grabbed air |
| Perception retry cooldown | v5 | no-blob retries burning all credits |
| Shape/color/z-band/fill quadruple filter | v6 | 12/12 aims hit furniture, not toys |
| Empty-close re-descend + reach_h 0.85 | v7 | empty closes at hover height |
| Explore freeze (dead-joint guard) | v8 | infinite explore loops on dead joints |
| Base approach phase (drive to target) | v9 | all targets beyond 0.85 m arm reach |
| Drive stability (gain/miss-limit/jump gate) | v9.1 | perception loss mid-drive |
| Fumble rescue (eef-gate escalation) | v9.2 | VLA fumbling at target, harness blocked |
| Head-pan servo (trunk joint 4 yaw) | v10 | target slid out of FoV while driving |
| Toy z-band priority in candidate ranking | v10.1 | furniture edges beat small toys |

## The two-loop tuning architecture

- **Loop level (within an episode)** — `hover_adapt`: every empty close lowers
  the pre-grasp hover offset by 0.04 m automatically; resets next episode.
- **Plan level (across episodes)** — `tools/tune_from_run.py` reads a run's
  jsonl event log + q_score, attributes failures (aim targets vs ground truth,
  VLA stage deadlocks, servo stall profiles), and emits suggested knob changes
  which are then frozen into a per-task profile via `--freeze`.

Configuration is three-layered: `DEFAULT_CONFIG` ← base json
(`configs/r9_harness_config.json`) ← per-task profile (`configs/profile_task7.json`,
selected by `task_id`).

## Results (honest attribution)

Every run's score was cross-checked against the harness event log and (where
available) ground-truth telemetry. **All 14 scores came from the VLA itself —
the harness has not yet completed a real toy grasp**, although v10 was the first
version to complete a full approach chain (drive + head-pan, 0.68/-0.83 m in
3.3 s, `approach_arrived`), and its in-reach aim converged 0.39 → 0.09 m.

| run | a | b | c | d | e | f | g | h | i | j | k | l | m | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_score | .333 | .167 | .5 | .333 | .167 | .667 | .333 | 0 | 0 | .5 | .167 | .167 | .5 | (v10.1) |
| harness | v1 | v2 | v2 | v3 | v4 | v5 | v6 | v7 | v8.5 | v9 | v9.1 | v9.2 | v10 | v10.1 |

Bare-policy baseline on the same instance/seed: {0.333, 0.667, 0.333}, mean 0.444.
Intervened mean 0.295 (13 runs) — dominated by VLA variance (0–0.667), not by
harm from the harness (early v1/v2 exceptions are documented in the log).

## Repo layout

```
r9_harness.py               core harness (single file, stdlib + numpy + scipy only)
mock_test_v9.py              offline test suite (A–F): synthetic camera, full chain
eval_b1k_wrapper.r9.py       integration point (how the wrapper calls the harness)
telemetry_wrapper.py         ground-truth telemetry collector (debug/forensics)
configs/                     base config + frozen per-task profiles
scripts/                     sim/policy container bring-up (eval server)
tools/                       run forensics, calibration, plan-level tuner
docs/                        design notes, iteration log, Q-matrix reference
```

## Quick start

```bash
# offline: verify the full intervention chain on synthetic scenes
python mock_test_v9.py

# server: bring up policy + sim (see scripts/ for the eval-server layout)
bash scripts/policy_up_r9.sh
bash scripts/run_sim_r9.sh picking_up_toys 0 <run-name>

# forensics on a finished run
python tools/analyze_episode.py harness_log.jsonl
python tools/tune_from_run.py harness_log.jsonl --score 0.167
```

## Key discoveries about the B1K stack (documented so you don't re-derive them)

1. **Base actions are normalized velocity commands** (body frame), gains ≈
   ax 0.575 / ay 0.777 m/s per unit; state `st[0:3]` is base *velocity*, not
   position (calibrated in `tools/calibrate_base.py`).
2. **The head can pan**: URDF `torso_joint4` = action dim 6 is a pure yaw joint
   (±3.05 rad) carrying the zed camera. The harness servos it during approach.
3. **True toys live at z ∈ [0.50, 0.81]** in this scene; warm-colored furniture
   edges appear at z ≈ 0.33–0.94 and dominate size-based candidate ranking.
4. `--instance-indices` in the evaluator takes the *index* into
   `TEST_INSTANCE_IDS` (301 = index 0), not the instance id.
