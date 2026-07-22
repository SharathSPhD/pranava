#!/usr/bin/env bash
# GB10: precompute Parakeet feats for the bi_instruct TTS clips, then broaden the live instruct model
# with translate + language-ID (warm-start instruct_ckpt.pt, save-bar protected). Self-healing chain.
set -u
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }
IMG=prabhasa/nemo-gb10:26.02
MOUNT="-v /home/sharaths/projects:/work -w /work -e PYTHONPATH=/work/pranava/src -e HF_HOME=/work/pranava/.hf_cache"

LOG "=== precompute feats: bi_instruct ==="
docker run --rm --name pranava-feats-biinstruct --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  $MOUNT $IMG python /work/pranava/scripts/alm/precompute_feats_dir.py data/alm/bi_instruct
LOG "feats rc=$?"

LOG "=== broaden instruct model (translate + language-ID) ==="
docker run --rm --name pranava-train-biinstruct --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  $MOUNT $IMG python /work/pranava/scripts/alm/train_bi_instruct.py 4
LOG "BI_INSTRUCT_rc=$?"
cat /home/sharaths/projects/pranava/data/alm/bi_instruct_metrics.json 2>/dev/null | head -40
LOG "BI_INSTRUCT_CHAIN_DONE"
