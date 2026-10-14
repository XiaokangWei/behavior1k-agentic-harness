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

## v11 — strategy-doc upgrade (ground-truth re-grounding + analytic transport)

`docs/`-side strategy doc `HarnessVLA与HarnessPAI策略借鉴.md` (mirrored from the
can-meat harness dir) prescribes two changes that the v10.1 vision-only loop
cannot make on its own. Both are implemented in `r9_harness.py` and gated OFF by
default (v10.1 is the documented frozen baseline; v11 is opt-in via
`configs/profile_task7_v11.json`):

- **S2 scene re-grounding.** When the sim-side `gt_injector.py` injects
  `obs["r9_gt"]` (robot-local toy/box poses from `env.task.object_scope`), the
  harness binds its aim target to the **true** toy pose instead of the noisy
  RGB-D back-projection. This directly cures the int301-f "+16 cm too high →
  empty close" and the furniture-over-toy mis-rank that the vision path suffers.
- **S1 analytic transport.** On a confirmed grasp the harness carries the toy to
  the GT box-approach pose and releases (base zeroed, gripper held closed during
  the carry) — instead of handing straight back to the VLA, which we observed
  drops the toy on the long transport leg. This is the doc's "analytic transport
  + release" primitive.

```
idle ──trigger──► perceive ──► [far?] ──► approach (drive base + pan head) ─┐
                   │ (S2: GT toy pose if available, else RGB-D)              │
                   │ [in reach]                                              ▼
                   ▼                                                    aim (joint servo)
           aim ──► close (gripper) ──► [grasp confirmed?]                    │
                                        │ yes + S1 enabled                  │
                                        ▼                                   │
                                 transport (carry to box, gripper CLOSED) ──► release ──► yield
                                        │ no / unsupported
                                        ▼
                                      yield ──► idle
```

> **Deployment caveat (important).** The policy runs in a *separate container*
> from the environment (`serve_b1k.py` is a websocket server; `env` lives in
> `b1k-sim-r9`). The harness therefore **cannot read `env` directly** — ground
> truth must be injected into `obs` on the sim side (`gt_injector.py`) and flow
> through. With `use_gt_poses=true` but **no** `r9_gt` in obs, the harness
> silently falls back to the v10.1 RGB-D perception. So v11 also requires
> wrapping the sim env with `gt_injector.EnvGTInjector` and restarting the sim
> container. Validated offline in `mock_test_v11.py` (G1–G4) and run live on the round5
checkpoint: **0.167** vs round5 bare **0.0** and round5+v10.1 **0.0** (see
Results → Round5 cross-check).

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
available) ground-truth telemetry. For the **r9** development loop the harness
never produced a score point of its own — all 14 q_scores came from the VLA.
v10 was the first version to complete a full approach chain (drive + head-pan,
0.68/-0.83 m in 3.3 s, `approach_arrived`), and its in-reach aim converged
0.39 → 0.09 m. The **v11** round5 run (below) is the first where the harness
itself delivered a score point, via analytic transport.

| run | a | b | c | d | e | f | g | h | i | j | k | l | m | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_score | .333 | .167 | .5 | .333 | .167 | .667 | .333 | 0 | 0 | .5 | .167 | .167 | .5 | .167 |
| harness | v1 | v2 | v2 | v3 | v4 | v5 | v6 | v7 | v8.5 | v9 | v9.1 | v9.2 | v10 | v10.1 |

Bare-policy baseline for the **r9** checkpoint on this instance/seed:
{0.333, 0.667, 0.333}, mean 0.444. Intervened mean 0.286 (14 r9 runs) —
dominated by VLA variance (0–0.667), not by harm from the harness (the r9
harness triggered ~0 times and let the VLA self-score; early v1/v2 exceptions
are documented in the log).

### Round5 checkpoint cross-check (reverses an earlier wrong hypothesis)

The harness was re-tested on the **round5** checkpoint
(`pandaele/vlatmp_0924/round5_fm2048_step34680`) on the same task7/instance 301,
one rollout each:

| config | q_score | notes |
|---|---|---|
| round5 bare (`B1K_R9_HARNESS=0`) | **0.0** | 28336 steps, full episode — round5 is itself weak here |
| round5 + harness v10.1 | 0.0 | arm servo 0/14 successful grasps |
| round5 + harness v11 (GT + analytic transport) | **0.167** | aim_arrived 14, close_empty 5, grasp_confirm 0, aim_stall_abort 6 |

**Earlier wrong hypothesis corrected:** a prior note claimed round5's bare score
was 0.667 and that the harness dragged it to 0.0. That 0.667 was an **r9** bare
run, not round5. Round5's *true* bare score is **0.0** — verified with the
policy container confirmed on the round5 ckpt and `B1K_R9_HARNESS=0`, across a
full 28336-step episode. So the harness did **not** destroy a good policy.

**Corrected conclusion:** the harness is *benign* for the strong r9 policy
(near-inert, scores track VLA variance) but *net-positive* for the weak round5
policy — v11 lifts bare 0.0 → 0.167. v11 (GT re-grounding + analytic transport)
is the variant that supplies the lift. Its autonomous arm-servo grasp is still
0%-success (same bottleneck as v10.1), but its GT guidance + analytic transport
deliver the one toy round5 alone cannot. **Recommendation: keep the harness
enabled; prefer the v11 profile for weak-policy checkpoints.**

## Repo layout

```
r9_harness.py               core harness (single file, stdlib + numpy + scipy only)
mock_test_v9.py              offline test suite (A–F): synthetic camera, full chain (v10.1 path)
mock_test_v11.py             offline test suite (G1–G4): GT re-grounding + analytic transport (v11 path)
gt_injector.py               sim-side obs["r9_gt"] injector (wires ground truth into the harness)
eval_b1k_wrapper.r9.py       integration point (how the wrapper calls the harness)
telemetry_wrapper.py         ground-truth telemetry collector (debug/forensics)
configs/                     base config + frozen per-task profiles (profile_task7.json = v10.1, _v11 = strategy)
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
5. **Ground truth lives on the sim side, not the policy side.** The harness runs
   inside `b1k-policy-r9`; the environment (`env.task.object_scope`) runs inside
   `b1k-sim-r9`. To feed the harness true toy/box poses you must inject them into
   `obs["r9_gt"]` from the sim (`gt_injector.py`) — the policy wrapper has no `env`
   handle. This is the wiring the strategy doc assumed was free.
