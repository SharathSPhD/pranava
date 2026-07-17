# Phase 5 — scaled to the fully-trained 1B Nemotron-H (on the RTX 5090)

The plan's endpoint: the Śabda-ALM now runs on the **fully-trained 1.13B-param Nemotron-H** core,
trained on the RTX 5090 (where prabhasa-samskrutam was trained). Executed remotely via the
`rtx5090-connect` skill.

## Setup
- Target: `ss@192.168.0.204` (RTX 5090, 32 GB), verified genuinely the 5090 before use.
- The **fully-trained 1B** is at `~/fusion-project/prabhasa-samskrutam/data/checkpoints/m4/final.pt`
  (1,129,634,752 params, 5003 steps / 5.25B tokens, loss 0.62→0.43, COMPLETED 2026-07-13) — far
  better than the local `m4_preflight`. It is the **baseline arm** (structured_channels=false).
- Environment: `prabhasa/nemo-5090:26.02` (Megatron-Core + mamba_ssm + NeMo ASR/TTS).
- `~/fusion-project/pranava/` created and the repo synced.

## The 1B injection point (make-or-break, PROVEN)
`src/pranava/alm/megatron_core.py:Megatron1BCore` reuses prabhasa's own `StructuredNemotronH` model
builder (so the architecture matches the checkpoint) and injects the Sphoṭa Projector's output via
Megatron's native **`decoder_input`** (`[s,b,h]` embeddings; MambaModel skips its internal
embedding). Verified end-to-end at 1B: `embed_tokens (1,T,1536)`, `forward_embeds → (1,T,256)`,
`features_embeds → (1,T,1536)`, greedy decode. Same surface as the small `SanskritCore`, so the
projector and the Sphoṭa-Lens port unchanged.

## Result — scaling clearly helps
Same stage-1 recipe (frozen 1B core + frozen Parakeet, train only the projector), 6 epochs on the
5090:

| core | train loss | val CER (audio) | vs no-audio (1.0) | Δ over baseline |
|---|---|---|---|---|
| 200M m2 (Phase 2) | 1.57→0.65 | 0.774 | beats | 0.226 |
| **1.13B m4 (Phase 5)** | **1.22→0.38** | **0.571** | beats | **0.429** |

The 1B core starts lower, descends faster, reaches **CER 0.571 vs 0.774**, and nearly **doubles**
the improvement-over-baseline. Scaling the Sanskrit substrate is the clear lever for ALM
understanding — as expected, and now measured.

## Honest notes
Still a coarse ASR-style CER (approximate TTS, 542 train items, 6 epochs); the win is the *relative*
scaling gain, cleanly measured. The 1B is the baseline arm (no kāraka-role channels); a
treatment-arm 1B (structured/aux heads) is future work. Projector-only training keeps the 1B core
frozen; unfreezing (LoRA) is the next lever.

## Artifacts
`data/alm/p5_metrics.json`, `projector_1b.pt` (fetched to the DGX); training on the 5090 under
`~/fusion-project/pranava/`. gate_P5 (dual) PASS.
