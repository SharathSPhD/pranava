# E5 — H-PROSODY: results & honest reading

Pre-registration: `research/prereg/H-PROSODY-E5.md` (committed before this analysis). Proposed by
the autoresearch loop (X0). Artifact: `data/experiments/e5_results.json`. Reproduce:
`python scripts/e5_prosody.py`.

## What was tested
80 sentences × 2 prosodic variants (neutral / urgent = rate ×1.3 + pitch +2 st, deterministic
librosa manipulation of the same TTS token stream). A grouped-CV linear probe (hold out whole
sentences, 3 seeds) decodes the prosody class from WavLM frames (speech) vs GPT-2 (text).

## Confirmatory results
| hypothesis | result |
|---|---|
| **PR1** speech carries prosody (acc > 0.5) | **acc = 1.000** ✔ |
| **PR2** text is prosody-blind (acc ≈ 0.5) | **acc = 0.500** (exact chance) ✔ |
| gap = acc_speech − acc_text | **0.500**, CI [0.425, 0.581], p≈0 |

Both pre-registered hypotheses hold. The **pragmatic-information gap is 0.5**: the speech encoder
perfectly recovers a prosodic contrast that the text model, seeing identical strings, cannot
represent at all. This is the docs' "acoustic blindness" made into a measured quantity — though
note (below) that PR2 is *definitional*: identical text cannot distinguish the variants, so text at
chance is guaranteed, not discovered. The contribution is the quantification, not the direction.

## Exploratory hypothesis — NOT supported (honest negative)
Pre-registered exploratory claim: prosody should be decodable *earlier* (more acoustic layers) than
meaning (which E2 localised to layer 9). **This is not supported.** Prosody decodes at **1.000 in
every one of the 13 layers** (layer sweep in `e5_results.json`). The rate+pitch manipulation is such
a strong, globally-consistent acoustic transform that it is trivially linearly separable at every
depth — it saturates. So E5 **cannot** localise prosody or compare its depth-profile to meaning's.

## What this does and does not establish
- **Does**: quantifies the speech-vs-text pragmatic gap (0.5) for a controlled prosodic contrast,
  and confirms the probe/pipeline behave correctly (text exactly at chance; speech well above).
- **Does not**: demonstrate *subtle* or *natural* prosody encoding, nor any layer localisation —
  the manipulation is too strong and uniform. The interesting scientific question (do speech
  encoders hold prosody and meaning at *different* depths?) needs a **subtler design**: natural
  expressive speech, or graded/small prosodic perturbations near the decoding threshold.

## Design lesson → next loop iteration
This near-ceiling saturation is itself the finding that should steer the autoresearch loop: the
prosody candidate should be *refined* (graded perturbation magnitudes; natural expressive corpus)
rather than repeated as-is. Recorded to the ledger as a completed-but-limited observation so the
selector down-weights a naïve repeat and favours the subtler variant.

## Threats
Synthetic prosody ≠ natural prosody; urgency = rate+pitch only; single speaker; WavLM-base only;
perfect separability makes CIs uninformative about *degree*.
