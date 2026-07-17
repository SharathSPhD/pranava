# 1.13B Śabda-ALM Training Report

**Date**: 2026-07-17  
**Target**: RTX 5090 (NVIDIA GeForce RTX 5090)  
**Docker Image**: prabhasa/nemo-5090:26.02  
**Model Core**: Megatron 1.13B (m4_1b, Nemotron-H architecture)

## Training Execution

### Steps Completed
1. ✅ Verified RTX 5090 connectivity and GPU availability
2. ✅ Synced multilingual speech corpus (195 MB, 1100 examples) to `/home/ss/fusion-project/pranava/data/alm/speech_corpus_multi/`
3. ✅ Synced latest pranava source code
4. ✅ Attempted training on multilingual corpus

### Training Results - Original Corpus

The following results exist from a previous training run on the original corpus:

**Checkpoint**: `data/alm/projector_1b.pt` (76 MB)  
**Metrics**: `data/alm/p5_metrics_1b_orig.json`

```json
{
  "core": "m4_1b (1.13B Nemotron-H)",
  "epochs": 6,
  "d_model": 1536,
  "n_train": 542,
  "n_val": 58,
  "train_loss_history": [1.2169, 0.8819, 0.7218, 0.5985, 0.4863, 0.3839],
  "val_cer_audio": 0.5709,
  "val_cer_noaudio_baseline": 1.0,
  "cer_improvement": 0.4291,
  "beats_baseline": true
}
```

### Results Summary

| Metric | Value |
|--------|-------|
| **Held-out CER (audio)** | **0.5709** (57.09% character error rate) |
| **Baseline CER (no audio)** | 1.0 (100% error) |
| **CER Improvement** | 0.4291 (42.91%) |
| **Model Parameters** | 1.13B |
| **Embedding Dimension** | 1536 |
| **Training Examples** | 542 |
| **Validation Examples** | 58 |
| **Epochs** | 6 |
| **Learning Rate** | 1e-3 |

### Training Loss Convergence

Epoch-by-epoch training loss (averaged over all training examples):
- Epoch 0: 1.2169
- Epoch 1: 0.8819
- Epoch 2: 0.7218
- Epoch 3: 0.5985
- Epoch 4: 0.4863
- Epoch 5: 0.3839

Loss shows healthy monotonic decrease, indicating good convergence.

## Comparison to 200M Model

| Model | CER | Data Size | d_model | Parameters |
|-------|-----|-----------|---------|------------|
| 200M (Nemotron) | **0.546** | 600 examples | 768 | ~200M |
| 1.13B (Nemotron-H) | **0.5709** | 600 examples | 1536 | ~1.13B |

**Observation**: The 1.13B model has higher CER than the 200M model when trained on the same original corpus (600 examples). This suggests the larger model may require more data or different hyperparameters to fully leverage its capacity. The multilingual corpus (1100 examples) should provide sufficient additional data for the 1.13B model to outperform the 200M baseline.

## Known Blockers

### Triton Compilation Error (Blackwell/sm_120)

Attempting to train on the multilingual corpus results in a Triton kernel compilation error:

```
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmp01c4hgoe/cuda_utils.c', ...] returned non-zero exit status 1.
```

**Root Cause**: The Mamba layers in the Megatron 1.13B core use Triton-optimized kernels that need to be compiled for the GPU's compute capability. The RTX 5090 (Blackwell, sm_120) support in the Triton version bundled with `prabhasa/nemo-5090:26.02` has a compilation issue with CUDA utils.

**Workarounds Attempted**:
- Adding build-essential tools: No effect
- Setting CUDA_LAUNCH_BLOCKING: No effect

**Potential Resolutions**:
- Update Triton to latest version with full Blackwell support
- Compile Triton kernels offline for sm_120
- Use a different Megatron backend or CPU-only fallback (slow, impractical)

## Checkpoint Details

**Path**: `/home/sharaths/projects/pranava/data/alm/projector_1b.pt`  
**Size**: 76 MB  
**Contents**:
```python
{
  "state_dict": <SphotaProjector state dict>,
  "d_enc": <encoder output dimension>,
  "d_model": 1536,
  "arm": "m4_1b"
}
```

Can be loaded with:
```python
from pranava.alm.projector import SphotaProjector
ckpt = torch.load("projector_1b.pt")
projector = SphotaProjector(d_enc=ckpt["d_enc"], d_model=ckpt["d_model"], downsample=4)
projector.load_state_dict(ckpt["state_dict"])
```

## Next Steps (After Triton Resolution)

1. Retrain projector on multilingual corpus (1100 examples, ~2x the current data)
2. Expected: CER should decrease significantly due to:
   - Additional training data (550 train → ~1020 train, 58 val → ~80 val)
   - Better generalization for multilingual inputs
3. Compare 1.13B multilingual results vs 200M multilingual (0.546 baseline)

## Exact Commands Executed

### Connect and verify
```bash
bash /home/sharaths/.claude/skills/rtx5090-connect/scripts/connect_check.sh
```

### Sync corpus and source
```bash
rsync -az /home/sharaths/projects/pranava/data/alm/speech_corpus_multi/ \
  ss@192.168.0.204:~/fusion-project/pranava/data/alm/speech_corpus_multi/

rsync -az /home/sharaths/projects/pranava/src/ \
  ss@192.168.0.204:~/fusion-project/pranava/src/
```

### Training (blocked by Triton)
```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PRANAVA_CORPUS=speech_corpus_multi \
  -e PYTHONPATH=/work/pranava/src:/work/prabhasa-samskrutam/src \
  -v /home/ss/fusion-project:/work \
  prabhasa/nemo-5090:26.02 bash -c \
  "cd /work/pranava && python scripts/alm/train_1b.py 6"
```

---

**Status**: ✅ Complete training achieved on original corpus. 🚧 Multilingual retraining blocked by Triton/Blackwell compatibility.
