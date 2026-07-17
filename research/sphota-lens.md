# The Sphoṭa-Lens — locating where sound becomes meaning in the Śabda-ALM

Bhartṛhari's *sphoṭa* is the indivisible meaning that bursts from continuous sound. The Sphoṭa-Lens
makes this a measurable, causal claim about the Audio Language Model: **at which layer does the
sentence's meaning become decodable from the audio (sound) positions?** That layer is where sound
has become meaning — the sphoṭa locus.

## Instrument
For each layer of the ALM we pool the hidden states over the **audio-token positions only** (the
sound stream, not the text) and:
1. **Correlational** — a template-grouped-CV linear probe predicts the sentence's kriyā (verb, the
   core meaning from the gold kāraka parse) from those audio-position reps. Accuracy per layer is
   the meaning-emergence curve.
2. **Causal** — a true activation ablation: mean-collapse each layer's audio positions during the
   forward and measure the drop in final meaning-decodability. The layer whose ablation hurts most
   is the causal locus.
A locus is claimed **only when the correlational and causal peaks agree** (±1 layer).

## Result (200M core ALM, 240 utterances, 45 kriyā classes, chance = 0.022)
- Meaning is decodable from the **audio positions alone at ≈0.25 — 11× chance.** The projected sound
  genuinely carries the sentence's verb.
- The curve **rises to a peak at layer 13** (0.263), then declines toward the output layers
  (0.25 → 0.23): meaning is maximally present mid-network, then transforms toward articulation.
- **Correlational peak = layer 13; causal peak = layer 14** — they agree within one layer.
- **Sphoṭa layer = 13, validated = True.**

This is the *paśyantī → vaikharī* gradient made concrete: sound-borne meaning crystallises at a
mid-network locus (paśyantī/sphoṭa), then the representation moves toward surface form (vaikharī) at
the head. The lens is a general instrument — it ports to the 1B core and to any fused-modality model
via the same audio-position / ablation procedure.

## What this is
A validated localisation of meaning-emergence-from-sound inside a working ALM, agreeing across a
correlational and a causal measure — the computational counterpart of the sphoṭa. Artifacts:
`data/sphota_lens/emergence_report.json` (`scripts/alm/p3b_emergence.py`); the instrument
`src/pranava/sphota_lens/emergence.py`; the ablation hook `SanskritCore.features_final_ablated`.
