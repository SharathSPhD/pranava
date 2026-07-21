#!/usr/bin/env bash
# GB10: after the EN baseline evals free the GPU, retry the two failed Shrutilipi-sa baselines
# (voxtral: zero-length-audio crash, now guarded; sushrota: abstract-class restore, now concrete)
# via the canonical eval_public.py, then re-score the board.
set -u
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }
while :; do
  ls /home/sharaths/projects/pranava/data/benchmark/librispeech_records/Voxtral* >/dev/null 2>&1 && break
  systemctl --user is-active --quiet pranava-en-chain.service || break
  sleep 180
done
LOG "EN evals done (or chain ended); retrying sa baselines"
for M in voxtral sushrota; do
  LOG "=== sa retry: $M ==="
  docker run --rm --name pranava-retry-$M --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/sharaths/projects:/work -w /work \
    -e PYTHONPATH=/work/pranava/src -e HF_HOME=/work/pranava/.hf_cache \
    prabhasa/nemo-gb10:26.02 python /work/pranava/scripts/alm/eval_public.py --dataset shrutilipi --model $M
  LOG "sa retry $M rc=$?"
done
/home/sharaths/projects/pranava/.venv/bin/python /home/sharaths/projects/pranava/scripts/alm/eval_public.py --dataset shrutilipi --score >/dev/null 2>&1
LOG "SA_RETRIES_DONE"
