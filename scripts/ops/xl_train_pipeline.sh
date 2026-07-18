#!/usr/bin/env bash
# GB10 XL pipeline: wait for the TTS expansion to finish → merge corpus → precompute Parakeet
# feats → train the 200M XL (projector+LoRA, EOS-weighted) with the fair eval. Runs unattended
# under systemd (systemd-run --user --unit=pranava-xl-pipeline …). Everything is resumable.
set -u
ROOT=/home/sharaths/projects/pranava
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

LOG "waiting for pranava-tts-expand.service to finish"
while systemctl --user is-active --quiet pranava-tts-expand.service; do sleep 60; done
LOG "TTS done; shard lines: $(wc -l < $ROOT/data/alm/speech_corpus_indic_xl/shard_000000.jsonl)"

LOG "merging corpus"
$ROOT/.venv/bin/python $ROOT/scripts/alm/expand_corpus_indic.py --merge || exit 1

LOG "precomputing Parakeet feats (container)"
$ROOT/scripts/alm/in_container.sh python /work/pranava/scripts/alm/precompute_feats_xl.py || exit 1

LOG "training 200M XL (3 epochs, EOS-weighted, fair eval)"
$ROOT/scripts/alm/in_container.sh python /work/pranava/scripts/alm/train_xl.py 3
rc=$?
LOG "train_xl finished rc=$rc"
[ -f $ROOT/data/alm/xl_metrics.json ] && cat $ROOT/data/alm/xl_metrics.json
exit $rc
