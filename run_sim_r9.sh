#!/bin/bash
# run_sim_r9.sh <task> <instances(csv)> <run_name> [extra eval args...]
# Round9 model runner: RGBDFullResWrapper by default (720/480 + depth).
# 224 direct-render (DefaultWrapper) is fault injection per 分辨率归因修正.md — never use for baselines.
# env knobs: ES (early stop), PRIV_DIAG (1=enable privileged diagnosis), WRAPPER (override)
TASK=$1; INST=$2; RUN=$3; shift 3
BASE=/home/bh1k/wsp/robot/behavior_1k
OUT=$BASE/inference/docker/out/$RUN
mkdir -p "$OUT"
docker rm -f b1k-sim-r9 >/dev/null 2>&1
exec docker run --rm --name b1k-sim-r9 \
  --network b1k_b1knet \
  --gpus '"device=0"' \
  --user 1000:1000 \
  --shm-size=8g \
  -e OMNIGIBSON_HEADLESS=1 -e OMNIGIBSON_GPU_ID=0 -e HOME=/tmp -e USER=bh1k -e LOGNAME=bh1k \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  -e B1K_EARLY_STOP=${ES:-0} -e B1K_EARLY_STOP_WINDOW=3000 -e B1K_EARLY_STOP_MIN_STEPS=4000 \
  -e B1K_EARLY_STOP_EPS=1e-6 -e B1K_TRAJ_SNAPSHOT_INTERVAL=1000 -e B1K_TRAJ_ROOT=/out/traj \
  -e PYTHONPATH=${EXTRA_PYTHONPATH:-/diag}:/opt/scienceflow/ScienceFlow-Robot/src:/opt/scienceflow/InquiryCraft/src \
  -e PRIV_DIAG_PATH=${PRIV_DIAG_PATH:-/out/priv_diag.jsonl} \
  -e B1K_TEL_PATH=${B1K_TEL_PATH:-/out/telemetry.jsonl} \
  -e B1K_TEL_EVERY=${B1K_TEL_EVERY:-1000} \
  -v $BASE/BEHAVIOR-1K/datasets:/data:ro \
  -v $BASE/inference/docker/sim/cache:/cache \
  -v /home/bh1k/wsp/robot/harness_exp:/diag:ro \
  -v "$OUT:/out" \
  scienceflow-b1k-runner:git-154e724bb165 \
  python -m omnigibson.eval.eval --task-name "$TASK" --host policy --port 8000 \
    --mode ${MODE:-public_test} \
    --instance-indices $INST --num-rollouts 1 \
    --env-wrapper ${WRAPPER:-omnigibson.eval.wrappers.RGBDFullResWrapper} \
    --output-dir /out "$@"
