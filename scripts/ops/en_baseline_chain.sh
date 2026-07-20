#!/usr/bin/env bash
# GB10 follow-on: after the Sanskrit baseline chain → English baselines on LibriSpeech test-clean
# (whisper/qwen/voxtral; sushrota is sa-only by design) → bilingual instruct TTS build (host GPU).
set -u
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }
while systemctl --user is-active --quiet pranava-baseline-chain.service; do sleep 120; done
LOG "sanskrit baselines done; starting ENGLISH baselines (LibriSpeech test-clean)"
for M in whisper qwen voxtral; do
  LOG "=== en baseline: $M ==="
  docker run --rm --name pranava-eval-en-$M --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/sharaths/projects:/work -w /work \
    -e PYTHONPATH=/work/pranava/src -e HF_HOME=/work/pranava/.hf_cache \
    prabhasa/nemo-gb10:26.02 python /work/pranava/scripts/alm/eval_public.py --dataset librispeech --model $M
  LOG "en baseline $M rc=$?"
done
LOG "english baselines done; building bilingual instruct TTS corpus"
/home/sharaths/projects/pranava/.venv/bin/python /home/sharaths/projects/pranava/scripts/alm/build_bi_instruct.py --n 1500
LOG "EN_CHAIN_ALL_DONE"
