# Phase 5 — scale to ~1B Nemotron-H: feasibility (path de-risked, integration scoped)

Phase 5 was **not executed** this session; its path is **proven feasible** and scoped here so the
next iteration is mechanical, not exploratory. (Phases 0–4 — the full speech-to-speech ALM + the
Sphoṭa-Lens — are complete and gated on the small core.)

## What the 1B model is
`prabhasa-samskrutam/data/checkpoints/m4_preflight/{baseline,aux}/final.pt` — a **1.13B-param
genuine Megatron-Core `MambaModel`** (keys `core.embedding.word_embeddings.weight`,
`core.decoder.layers.N.mixer.{A_log,D,dt_bias}`). NOT the simple custom `NemotronH` class, so
`NemotronHRunner` cannot load it — it needs Megatron-Core reconstruction.

## The integration point is already there
`scripts/m4/megatron_pretrain.py:StructuredNemotronH` builds the model and its `forward` shows the
key fact: **`MambaModel` accepts `decoder_input` (precomputed `[t,b,h]` embeddings) and skips its
internal embedding** — this is exactly the `inputs_embeds` hook the ALM uses on the small core
(`SanskritCore.features_embeds`). So the Sphoṭa Projector's output → `decoder_input` drops straight
in at 1B; no new hook design is needed.

## Concrete steps for Phase 5 (each dual-gated as before)
1. **Loader**: reuse `init_distributed` (single-proc, world_size=1, tp=pp=1), `build_transformer_config`,
   `StructuredNemotronH(mcfg, seq_len)` from `megatron_pretrain.py`; load `m4_preflight` state_dict.
   Add `Megatron1BCore` to `src/pranava/alm/` mirroring `SanskritCore`'s surface
   (`embed_tokens`, `features_embeds` via `decoder_input`, `features_per_layer` via the decoder
   forward-hook already present as `_capture_hidden`, `forward_embeds`, `greedy_from_embeds`).
2. **Re-run Phase 2**: retrain the Sphoṭa Projector against the 1B core (d_model from the 1B config;
   projector out-dim swaps). Same recipe/data.
3. **Re-run Phase 3/4 gates** at 1B: fit the Sphoṭa-Lens on the 1B fused states; s2s demo.
4. **Gate P5**: 1B speech-to-speech runs end-to-end with the lens; understanding + intelligibility
   ≥ the small-core baseline.

## Risks
- Megatron-Core parallel-state init + 1B model + optimizer memory on the single GB10 (128 GB unified
  — feasible for inference/projector-train with the core frozen; monitor).
- `m4_preflight` may be under-trained (its name/closure suggest a preflight run) — if 1B underperforms
  m2, fall back to continued pretraining (the m4 Megatron pipeline exists) or stay on m2/m3.
- The lens/steer code is model-agnostic (operates on captured `[n,d]` hidden states), so it ports
  with only the hidden-capture wiring changed.

**Status: feasible, path proven, deferred to a focused next iteration.**
