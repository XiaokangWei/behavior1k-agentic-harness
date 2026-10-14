#!/usr/bin/env python3
"""Init the b1k-r9-harness git repo via dulwich (no Xcode CLT needed)."""
import os

# sandbox mkdir shim raises EEXIST as PermissionError; make mkdir idempotent
_orig_mkdir = os.mkdir
def _mkdir(p, mode=0o777, *, dir_fd=None):
    try:
        _orig_mkdir(p, mode, dir_fd=dir_fd)
    except (FileExistsError, PermissionError):
        import stat as _stat
        if not os.path.isdir(p):
            raise
os.mkdir = _mkdir

from dulwich import porcelain

ROOT = "/Users/xiaokangwei/WorkBuddy/2026-10-13-01-29-05"
os.chdir(ROOT)
repo = porcelain.init(".")

include = []
for f in sorted(os.listdir(ROOT)):
    if not os.path.isfile(f):
        continue
    if f == ".DS_Store":
        continue
    include.append(f)
for f in sorted(os.listdir("configs")):
    include.append(os.path.join("configs", f))

porcelain.add(repo, include)
msg = (
    "b1k r9 task-level harness v7: watchdog + single-aim intervention\n"
    "\n"
    "Core:\n"
    "- r9_harness.py: node-level watchdog (stage stall / empty closes / no-close\n"
    "  detectors), HSV toy perception with shape/fill/saturation/z-band filters\n"
    "  (v6), Broyden rank-1 online-jacobian EEF servo, bounded close phase with\n"
    "  descend-retry on empty close (v7), per-task frozen profiles, hot-reload\n"
    "  config, loop-level hover adaptation\n"
    "- telemetry_wrapper.py: sim-side instrumented wrapper (camera dump, per-object\n"
    "  wpos/lpos, per-predicate satisfaction, depth/rgb patch sampling)\n"
    "- eval_b1k_wrapper.r9.py: harness hook into the policy eval loop\n"
    "\n"
    "Tuning/analysis:\n"
    "- tune_from_run.py: cross-episode tuner -> suggests knobs, freezes profiles\n"
    "- validate_aims_gt.py: aim-target ground-truth check vs telemetry lpos\n"
    "- analyze_episode.py / analyze_aims.py / analyze_int_run.py: run forensics\n"
    "- toy_colors.py / tune_convention.py / discriminate_depth.py /\n"
    "  validate_projection.py / validate_blobs.py: perception calibration suite\n"
    "\n"
    "Ops: policy_up_r9.sh / policy_up_r5.sh / run_sim_r9.sh\n"
    "Configs: configs/r9_harness_config.json + configs/profile_task7.json\n"
    "  (v7: reach_h_max 0.85, close_retries 2, descend_step 0.06, servo_tol 0.07)\n"
    "\n"
    "Task 7 (picking_up_toys, instance 301) score history on round9 ckpt:\n"
    "  bare runs (observe): 0.333 / 0.667 / 0.333 (mean 0.444)\n"
    "  harness v1-v7:       0.333 / 0.167 / 0.500 / 0.333 / 0.167 / 0.667 / ...\n"
)
porcelain.commit(
    repo,
    message=msg,
    author=b"xiaokangwei <xiaokangwei@bh1k-dev.local>",
    committer=b"xiaokangwei <xiaokangwei@bh1k-dev.local>",
)
print("committed files:")
for f in include:
    print(" ", f)
print("HEAD:", repo.head())
