"""Sim-side ground-truth injector for the R9 harness (Harness VLA S2 re-grounding).

WHY THIS FILE EXISTS
--------------------
The strategy doc (HarnessVLA与HarnessPAI策略借鉴.md, S1+S2) prescribes binding the
harness's aim/transport targets to the TRUE toy/box geometry ("scene re-grounding")
instead of chasing a noisy RGB-D blob or replaying a fixed displacement. That
requires ground-truth object poses -- but the policy runs in a SEPARATE container
from the environment (`serve_b1k.py` is a websocket server; `env` lives in
`b1k-sim-r9`). So GT cannot be read from `env` inside the harness. The fix is to
inject GT into `obs` on the SIM side and let it flow through the policy server.

This wrapper computes robot-local toy/box poses + per-toy `in_box` status from
`env.task.object_scope` (the same privileged state the offline `telemetry_wrapper`
reads) and writes `obs["r9_gt"]` every step. The harness consumes it when
`use_gt_poses=True`.

USAGE (sim driver, where the env is wrapped -- same slot as telemetry_wrapper):
    from gt_injector import EnvGTInjector
    env = EnvGTInjector(env)          # replaces/composes the env wrapper
Then launch the policy with B1K_R9_HARNESS=1 and a config that sets
use_gt_poses=true (and analytic_transport_after_grasp=true for the S1 carry).

NOTE: keep telemetry_wrapper in the stack too if you want the offline jsonl logs;
this injector only adds obs["r9_gt"] and does NOT log.
"""
import numpy as np
import torch as th

from omnigibson.eval.wrappers.rgbd_full_res_wrapper import RGBDFullResWrapper
from omnigibson.utils import transform_utils as T


def _lst(x):
    try:
        a = np.asarray(x, dtype=float)
        return a.reshape(-1).tolist() if a.size else None
    except Exception:
        return None


class EnvGTInjector(RGBDFullResWrapper):
    """Injects obs["r9_gt"] = {toys:[{name,lpos,lorn,in_box}], box:{lpos,lorn}}."""

    def __init__(self, env, inbox_proxy=0.22):
        super().__init__(env)
        self._n = 0
        self._inbox_proxy = float(inbox_proxy)

    def _robot(self):
        return self.env.robots[0]

    def _find_box_name(self):
        for name in self.env.task.object_scope.keys():
            if "box" in str(name).lower():
                return str(name)
        return None

    def _in_box(self, name, ent, box_name, box_lpos):
        # Authoritative first: evaluate the task's ground-goal predicates that
        # mention BOTH this toy and the box. Fall back to a geometric proxy.
        try:
            task = self.env.task
            for option in task.ground_goal_state_options:
                for pred in option:
                    s = str(pred)
                    if name in s and box_name is not None and box_name in s:
                        try:
                            return bool(pred.evaluate(task._evaluate_predicate))
                        except Exception:
                            pass
        except Exception:
            pass
        if box_lpos is not None:
            try:
                wpos, worn = ent.get_position_orientation()
                base_pos, base_orn = self._robot().get_position_orientation()
                lpos, _ = T.relative_pose_transform(wpos, worn, base_pos, base_orn)
                return float(np.linalg.norm(np.asarray(lpos, dtype=float)[:2]
                                            - np.asarray(box_lpos, dtype=float)[:2])) < self._inbox_proxy
            except Exception:
                pass
        return False

    def _build_gt(self, obs):
        row = {"toys": [], "box": None, "step": self._n}
        try:
            robot = self._robot()
            base_pos, base_orn = robot.get_position_orientation()
            box_name = self._find_box_name()
            box_lpos = None
            if box_name is not None and hasattr(self.env.task.object_scope[box_name], "get_position_orientation"):
                bwpos, bworn = self.env.task.object_scope[box_name].get_position_orientation()
                blpos, _ = T.relative_pose_transform(bwpos, bworn, base_pos, base_orn)
                box_lpos = _lst(blpos)
                row["box"] = {
                    "name": box_name,
                    "lpos": box_lpos,
                    "lorn": _lst(bworn),
                }
            for name, ent in self.env.task.object_scope.items():
                if not hasattr(ent, "get_position_orientation"):
                    continue
                if name == box_name:
                    continue
                wpos, worn = ent.get_position_orientation()
                lpos, lorn = T.relative_pose_transform(wpos, worn, base_pos, base_orn)
                rec = {
                    "name": str(name),
                    "lpos": _lst(lpos),
                    "lorn": _lst(lorn),
                    "in_box": self._in_box(str(name), ent, box_name, box_lpos),
                }
                row["toys"].append(rec)
        except Exception as e:
            row["err"] = repr(e)
        return row

    def step(self, action, n_render_iterations=1):
        obs, reward, terminated, truncated, info = super().step(
            action, n_render_iterations=n_render_iterations)
        self._n += 1
        try:
            obs["r9_gt"] = self._build_gt(obs)
        except Exception:
            pass   # never break the rollout
        return obs, reward, terminated, truncated, info
