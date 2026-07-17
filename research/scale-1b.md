# Scaling the Śabda-ALM to 1.13B — a genuine multilingual run on the RTX 5090

The 200M byte-core proved the architecture; the operator asked to "go with the 1.13b model" in
parallel. So the Sphoṭa Projector was trained against the **fully-trained 1.13B Megatron-Core
Nemotron-H** prabhasa core (`m4/final.pt`, d_model 1536) on the **same multilingual corpus** the
200M uses — a real scale-up on identical data, run on the RTX 5090 where prabhasa was trained.

## Result (held-out multilingual val, 108 items, overall CER ↓)
| model | core | adaptation | CER |
|---|---|---|---|
| **Śabda-ALM 1.13B** | 1.13B Nemotron-H (Megatron) | projector-only | **0.682** |
| Śabda-ALM 200M | 200M byte-core | projector + LoRA | 0.742 |
| no-audio baseline | 1.13B | — | 0.978 |

Training drove the loss 1.19 → 0.78 over 6 epochs; audio conditioning cuts CER from the 0.978
no-audio baseline to 0.682 (a 0.30 improvement — the projector genuinely conveys speech content to
the larger core).

## The honest reading
- **Scale helps.** On the identical multilingual corpus, the 1.13B core reaches 0.682 overall vs the
  200M's 0.742 — a real 0.06 gain from ~5.6× the parameters.
- **Not yet a clean adaptation-controlled comparison.** The 1.13B is *projector-only* while the 200M
  has *LoRA* on top; so this shows "bigger frozen core + trained projector" beats "small core +
  LoRA", but a LoRA'd 1.13B (the next step) is needed to separate scale from adaptation cleanly.
  Recall the earlier within-adaptation finding still stands: on the controlled corpus, 200M+LoRA
  (0.548) beat 1.13B projector-only (0.571) — adaptation matters as much as scale.
- **Reproducible.** `scripts/alm/train_1b.py` (PRANAVA_CORPUS=speech_corpus_multi) on the RTX 5090 in
  `prabhasa/nemo-5090:26.02`; artifacts `data/alm/{p5_metrics_1b_multi.json, projector_1b_multi.pt}`;
  leaderboard `data/benchmark/scale_1b_leaderboard.json`. Gated: `python gates/check.py SC`.
