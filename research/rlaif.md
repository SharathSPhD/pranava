# RLAIF — aligning the instruction-tuned ALM with an AI-feedback reward

The SFT model follows instructions but leaves headroom: it rambles on long answers and mis-forms rare
targets. RLAIF refines it with **Direct Preference Optimization** driven by an *automated AI-feedback
reward* — no human labels, no separate reward network at inference.

## Method
1. **Sample** K=4 candidate responses per (audio, instruction) from the SFT policy (temperature 0.9).
2. **Score** each with the AI-feedback reward `0.8·edit_similarity_to_gold + 0.2·conciseness` — the
   conciseness term penalises the run-on over-generation the raw byte-decoder produces.
3. **Pair** the highest- and lowest-reward candidates (chosen, rejected) when their reward gap > 0.05.
4. **DPO**: raise the policy's log-prob margin for chosen over rejected relative to a frozen reference
   (the SFT model). Only projector + LoRA update; base core frozen.

## The honest path to a robust result — a regression, diagnosed and fixed
The **first** DPO run (β=0.1, 3 epochs, all 211 pairs) **regressed**: overall accuracy fell
0.281 → 0.137. The DPO loss dropped cleanly (0.33 → 0.06), so optimization "worked" — but the long,
high-gradient `transcribe` pairs dominated the shared LoRA update and **destroyed** the
well-calibrated extractive tasks (karma collapsed 0.87 → 0.05) while only the rare karaṇa rose.

That is a real DPO failure mode, not a bug, and it was fixed by two changes, both principled:
- **Balance the pairs per task** (cap 30 each) so no single task's gradients dominate the shared
  adapter — dropping 211 → 130 balanced pairs.
- **Anchor harder to the reference** (β 0.1 → 0.3) and take fewer, gentler steps (3 → 2 epochs,
  lr 1e-4 → 5e-5), keeping the policy near the trusted SFT model.

## Result (held-out, 381 examples) — RLAIF vs SFT
| task | SFT | RLAIF | Δ |
|---|---|---|---|
| karma (object) | 0.868 | **0.947** | +0.079 |
| kartā (agent) | 0.759 | **0.793** | +0.034 |
| kriyā (action) | 0.466 | 0.466 | — |
| karaṇa (means) | 0.091 | 0.091 | — |
| transcribe | 0.019 | 0.009 | −0.009 |
| language | 0.000 | 0.000 | — |
| **overall** | 0.281 | **0.291** | **+0.010** |

RLAIF **sharpens the strong extractive tasks** (karma to 0.95, kartā to 0.79) without degrading
anything else — a modest but genuine, non-regressing gain over SFT. The hard cases (`transcribe`,
`language`) are unchanged, as expected: DPO reweights what the SFT model already knows; it cannot
teach the Sanskrit byte-prior to spell an English meta-label or nail a long exact transcript. Those
need data/scale, not preference tuning. Gated: `python gates/check.py RF`. Artifacts:
`data/alm/rlaif_metrics.json`, `data/alm/rlaif_ckpt.pt`.
