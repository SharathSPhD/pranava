#!/usr/bin/env bash
# Build the bilingual instruct corpus once the GB10 baseline retries free the GPU.
set -u
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }
while systemctl --user is-active --quiet pranava-sa-retry.service; do sleep 120; done
LOG "retries done; building bilingual instruct corpus (n=1500)"
/home/sharaths/projects/pranava/.venv/bin/python \
  /home/sharaths/projects/pranava/scripts/alm/build_bi_instruct.py --n 1500
LOG "INSTRUCT_BUILD_rc=$?"
wc -l /home/sharaths/projects/pranava/data/alm/bi_instruct/manifest.jsonl 2>/dev/null
