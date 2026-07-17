# Pre-registration — H-HOLISM-E7 (definitive, properly-powered)

Committed before E7 results. Settles the holism question after E6 falsified the underpowered E2
version. Matched-vocabulary minimal manipulation of resolution timing.

## Design
Same 8 verb classes appear as the meaning label in two positions, holding vocabulary constant:
- **late** pool = E6's 240 verb-final items (verb is the last word).
- **early** pool (new) = 240 items, same 8 verbs at position ~2 (e.g. "The captain inspected the
  cargo after the storm."), 8 templates × crossed slots, valid template-grouped CV.
A single linear probe is trained across the pooled early+late items (same 8-class label space,
grouped CV by template) and applied to cumulative representations → Holism Index per item.

## Confirmatory hypotheses (Holm across 2)
- **P1**: HI(late) > HI(early), within each modality (sanity: end-resolving → later decodability).
- **P2 (the real question)**: HI_speech(late) − HI_text(late) > 0 (speech more holistic than text
  on genuinely late-resolving items), now with proper power and valid CV. Null = 0.

## Stats
Grouped-CV probe, 3 seeds; Holism Index; bootstrap 2000 CI; Holm across the 2 tests. Fixed data.

## Decision
If P2's CI includes 0 → the E2 headline is definitively **not supported** (confirms E6). If P2's
CI excludes 0 in the predicted direction under this proper design → a real, rescued effect. Either
is reported. P1 is expected true and serves mainly as a design sanity check.

## Threats
Same TTS/single-speaker limits; early-position verbs may still carry weak context cues (mitigated
by crossing 8 templates × 6 objects). Speech/text tokenization granularity differs (relative
position normalises).
