#!/usr/bin/env bash
# Run a command inside the prabhasa/nemo-gb10 container with the projects tree mounted.
# The container has mamba_ssm + NeMo (ASR/TTS) + torch/CUDA + the prabhasa stack — everything
# the Śabda-ALM needs. Usage: scripts/alm/in_container.sh <cmd...>
set -euo pipefail
IMG="${ALM_IMAGE:-prabhasa/nemo-gb10:26.02}"
# NB: cwd is /work (NOT /work/pranava): mamba_ssm's triton JIT shells out to gcc, which would
# otherwise pick up pranava/specs/ as a gcc spec file ("cannot read spec file './specs'").
exec docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/sharaths/projects:/work -w /work \
  -e PYTHONPATH=/work/pranava/src:/work/prabhasa-samskrutam/src:/work/prabodha/src:/work/prabodha/vendor/jacobian-lens \
  -e HF_HOME=/work/pranava/.hf_cache \
  "$IMG" "$@"
