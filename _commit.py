"""Commit the reorganized repo with dulwich (no Xcode CLT on this Mac)."""
import os

_real_mkdir = os.mkdir
def _mkdir(p, *a, **k):
    if os.path.isdir(p):
        return
    return _real_mkdir(p, *a, **k)
os.mkdir = _mkdir

import fnmatch
from dulwich import porcelain
from dulwich.repo import Repo

ROOT = "/Users/xiaokangwei/WorkBuddy/2026-10-13-01-29-05"

ignore = []
for line in open(os.path.join(ROOT, ".gitignore")):
    line = line.strip()
    if line and not line.startswith("#"):
        ignore.append(line)

keep = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames
                   if d not in (".git", ".workbuddy", "__pycache__")]
    rel = os.path.relpath(dirpath, ROOT)
    for f in filenames:
        r = os.path.normpath(os.path.join(rel, f)) if rel != "." else f
        if any(fnmatch.fnmatch(r, p) or fnmatch.fnmatch(f, p) for p in ignore):
            continue
        keep.append(r)

keep.sort()
print(f"staging {len(keep)} files")

repo = Repo(ROOT)
porcelain.add(repo, [os.path.join(ROOT, k) for k in keep])
porcelain.commit(
    repo,
    b"Finalize v10.1 harness design: repo reorganization, README, iteration log\n\n"
    b"- r9_harness.py v10.1: toy z-band priority in candidate ranking (in-reach + far),\n"
    b"  servo_tol 0.10, head-pan servo, base approach, fumble rescue, explore-freeze\n"
    b"- docs/: design notes, full v1->v10.1 iteration log with per-run forensics,\n"
    b"  Q-matrix reference\n"
    b"- tools/: run forensics, base calibration, plan-level tuner (dual-loop arch)\n"
    b"- scripts/: eval-server bring-up (sim + policy containers)\n"
    b"- mock_test_v9.py: offline test suite A-F covering the full intervention chain\n",
    author=b"XiaokangWei <xiaokangwei@users.noreply.github.com>",
    committer=b"XiaokangWei <xiaokangwei@users.noreply.github.com>",
)
print("committed")
for e in repo.get_refs():
    print(e.decode())
