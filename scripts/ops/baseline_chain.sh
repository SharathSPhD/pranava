#!/usr/bin/env bash
# GB10: wait for the Whisper eval container to finish, then run the remaining Shrutilipi-sa test
# baselines sequentially (mms → qwen → voxtral → sushrota), each in its own container run.
set -u
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

LOG "waiting for pranava-eval-whisper to exit"
while [ "$(docker inspect -f '{{.State.Status}}' pranava-eval-whisper 2>/dev/null)" = "running" ]; do sleep 60; done
LOG "whisper done; chaining remaining baselines"

for M in mms qwen voxtral sushrota; do
  LOG "=== baseline: $M ==="
  docker run --rm --name pranava-eval-$M --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/sharaths/projects:/work -w /work \
    -e PYTHONPATH=/work/pranava/src -e HF_HOME=/work/pranava/.hf_cache \
    prabhasa/nemo-gb10:26.02 python /work/pranava/scripts/alm/eval_shrutilipi.py --model $M
  LOG "baseline $M rc=$?"
done
LOG "ALL BASELINES DONE"
ls -la /home/sharaths/projects/pranava/data/benchmark/shrutilipi_records/ 2>/dev/null
