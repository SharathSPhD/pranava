# Pre-registration — H-PROSODY (E5)

Committed before E5 results. Proposed by the autoresearch loop (EFE selector, X0).

## Question
The docs' "acoustic blindness" claim: text representations are structurally blind to prosody, while
speech representations carry it. Operationalise and *quantify* the pragmatic-information gap, and
localise where prosody lives in the speech encoder.

## Design (controlled, ground-truth-exact)
Take a fixed subset of E1 sentences. Render each in two **prosodic variants** by deterministic
acoustic manipulation of the same neutral TTS token stream:
- **neutral**: as synthesized.
- **urgent**: time-compressed (rate ×1.3) + pitch-raised (+2 semitones) via librosa.
The manipulation *is* the label, so ground truth is exact. **The text is identical across variants.**

## Metrics & hypotheses
- **PR1 (speech carries prosody)**: a linear probe decodes the prosody class from WavLM frames well
  above chance (0.5). Report accuracy with grouped-CV (hold out whole sentences).
- **PR2 (text is prosody-blind — structural check)**: the same probe on the text model is at
  chance, because the input is identical across variants. This is definitional and serves as a
  validity anchor, not a discovery; we report it to quantify the *gap* = acc_speech − acc_text.
- **Exploratory**: layer sweep — which WavLM layer best encodes prosody; is prosody decodability
  higher in *earlier* (acoustic) layers than meaning decodability (which E2 found peaks later)?

## Stats
Grouped-CV accuracy, 3 seeds, mean ± sd. Bootstrap 95% CI on the speech−text gap. Fixed dataset.

## Honesty
PR2 being trivially at chance is stated up front; the contribution is the *quantified* gap and the
*localisation* of prosody, not the (definitional) fact that identical text can't distinguish it.
A speech probe near chance would be a surprising negative (pipeline failure) and reported as such.

## Threats
Synthetic prosody via signal manipulation is not natural prosody; urgency here = rate+pitch only,
not the full acoustic signature. Localisation is model-specific (WavLM-base).
