#!/bin/bash
# Start (or replace) the policy server with the ROUND5 FM2048 checkpoint
# (round5_fm2048_step34680 -- best bare-run model on task 7: 0.667).
# Same dense FM2048 architecture as round9 -> reuse the round9 inference code
# via its --checkpoint override (see inference_code README: CHECKPOINT_DIR knob).
# knobs: XLA_MEM_FRACTION (default 0.6), PORT (default 8000)
R5ROOT=/home/bh1k/wsp/robot/behavior_1k/models/pandaele/vlatmp_0924
CODE=$R5ROOT/inference_code_fm2048_round9_step68203
CKPT=$R5ROOT/round5_fm2048_step34680
PORT=${PORT:-8000}

if [ ! -d "$CKPT/params" ]; then
  echo "ROUND5 CKPT NOT FOUND at $CKPT"; echo "Upload round5_fm2048_step34680 (params/ + assets/) first."; exit 1
fi

docker rm -f b1k-policy-r9 >/dev/null 2>&1
docker stop b1k-policy-hx  >/dev/null 2>&1          # free hostname alias "policy"
docker stop b1k-policy     >/dev/null 2>&1
sleep 1

docker run -d --name b1k-policy-r9 \
  --network b1k_b1knet --network-alias policy \
  --gpus '"device=0"' --shm-size=8g \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION:-0.6} \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  -e B1K_R9_HARNESS=${B1K_R9_HARNESS:-1} \
  -e B1K_R9_CONFIG=/logs/r9_harness_config.json \
  -v /home/bh1k/wsp/robot/behavior_1k/BEHAVIOR-1K/datasets:/data:ro \
  -v /home/bh1k/wsp/robot/behavior_1k/models:/models:ro \
  -v "$CODE:/app/r9code:ro" \
  -v /home/bh1k/wsp/robot/harness_exp/r9logs:/logs \
  --entrypoint bash \
  b1k-policy:champion \
  /app/r9code/run_round9.sh --port "$PORT" \
    --checkpoint /models/pandaele/vlatmp_0924/round5_fm2048_step34680

for i in $(seq 1 180); do
  if docker logs b1k-policy-r9 2>&1 | grep -q "Creating server"; then
    echo "POLICY R5 READY after $((i*10))s"; exit 0
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' b1k-policy-r9 2>/dev/null)" != "true" ]; then
    echo "POLICY R5 DIED"; docker logs b1k-policy-r9 2>&1 | tail -40; exit 1
  fi
  sleep 10
done
echo "POLICY R5 TIMEOUT"; docker logs b1k-policy-r9 2>&1 | tail -40; exit 1
