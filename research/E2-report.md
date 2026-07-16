# E2 — H-HOLISM: results & honest reading

Pre-registration: `research/prereg/H-HOLISM.md` (committed before this analysis, git a7afd70).
Artifacts: `data/experiments/e2_results.json`, `e2_trajectories.png`. Reproduce:
`python scripts/e2_run.py` (features cached under `data/experiments/features/`).

## What was tested
For 288 controlled utterances we measured, at each relative position, the decodability of the
sentence's meaning label from a **speech** model (WavLM-base, layer 9) and a **text** model
(GPT-2, layer 6), via a template-grouped-CV linear probe (3 seeds). The **Holism Index (HI)** =
fraction of total decodability gain arriving in the last 20% of the utterance (high = late
"flash", holistic; low = gradual accrual). This operationalises Vākyapadīya 2.143 — *pratibhā* as
the distinct sentence-meaning that arises when word-meanings are grasped together.

## Probe validity (sanity, not a result)
Full-utterance decodability ≫ chance (0.091 for 11 classes): **speech 0.47, text 0.59**. HI is
therefore computed on a probe that genuinely reads meaning. ✔

## Confirmatory results (Holm across the 3 tests actually run)
| test | HI effect (a−b) | 95% CI | one-sided p (pred. dir.) | supported |
|---|---|---|---|---|
| P1 speech: late>early | +0.010 | [−0.066, 0.080] | 0.396 | no |
| P1 text: late>early | −0.101 | [−0.157, −0.039] | 0.9995 | no (opposite direction) |
| **P2 speech>text on late** | **+0.093** | **[0.016, 0.167]** | **0.0095** | **YES** |

**Decision-table outcome: "P2 only".** The speech model resolves *late-resolving* sentences more
holistically (later-concentrated meaning gain) than the text model. Within-modality, resolution
timing did not move speech HI (P1-speech null); text was actually *less* holistic for late items
(it front-loads decodability from lexical priors), the opposite of the naïve prediction.

## The confound I checked — and it does NOT drive the effect
Garden-path items all share one label ("noun_is_verb") and a template frame, so their label could
be cued structurally rather than resolved semantically late. Exploratory HI by structure:

| structure | n | HI speech | HI text |
|---|---|---|---|
| canonical (early) | 144 | 0.204 | 0.284 |
| **garden_path** | 72 | **0.128** | **0.131** |  ← no speech advantage (confound is inert)
| **verb_final** | 48 | **0.406** | **0.169** |  ← strongest, cleanest signal
| verb_first (early) | 24 | 0.377 | 0.022 |

On **verb_final** items — where the meaning label *is* the final verb, so meaning genuinely
resolves at the last word — speech HI 0.41 vs text 0.17, **effect +0.237, CI [0.106, 0.361],
p≈0** (exploratory). The garden-path items, where the confound would live, show **no** difference
(0.128 vs 0.131). So P2 is carried by the items where late resolution is real, not by the
structurally-cued ones. This is the sphoṭa-consistent core of the finding.

## Honest reading
- **Supported, with a clean mechanism**: a continuous acoustic representation concentrates
  meaning-decodability *late* for sentences that genuinely resolve late (verb-final), whereas a
  discrete text model front-loads it. This is one concrete, reproducible sense in which speech
  representations are "more holistic" — consistent with the *pratibhā*/sphoṭa picture that
  sentence-meaning arrives as a late unified grasp rather than running word-by-word accrual.
- **Not supported**: the blanket claim that speech is more holistic *regardless* of item type
  (garden-path shows nothing), and that late-vs-early moves HI *within* speech (it doesn't — speech
  HI is moderate across the board except where resolution is structurally forced to the end).

## Threats to validity (not hidden)
Synthetic single-speaker TTS; small closed vocabulary (11 labels); word→frame alignment is
implicit in the relative-position grid; "cumulative mean-pool" is one operationalisation of
"grasped so far"; speech (~50 Hz frames) and text (BPE tokens) tokenize differently — relative
position normalises but does not equate them. GPT-2's strong lexical priors make its early
decodability high; a bidirectional or larger text model might behave differently. The verb_final
result is exploratory (not Holm-protected) and rests on 48 items. These bound the claim; the
verb_final contrast is the part most worth replicating at larger scale (a future loop).
