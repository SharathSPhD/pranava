# E6 — scaling the verb-final contrast CORRECTS the E2 finding (honest walkback)

Loop-proposed (X0: "scale verb-final to tighten the CI"). The scaling did not tighten a real
effect — it **revealed the effect was an artifact of the original stimulus design**. This is the
most important result in the project so far, and it is a negative. Reported in full.

## What E2/E2b/E3 claimed
On the E1 `verb_final` subset (48 items), speech encoders showed higher Holism Index than text
(WavLM 0.41 / HuBERT 0.43 / vs GPT-2 0.17 / distilgpt2 0.15), CIs excluding 0 — read as "speech
resolves verb-final meaning more holistically (late) than text."

## What E6 finds
A properly-crossed, well-powered verb-final set — **240 items, 8 templates × 6 objects × 8 verb
classes**, fully crossed so the final verb is NOT predictable from context, valid template-grouped
CV (8 groups) — shows **no speech advantage**:

| | HI speech | HI text | effect | CI95 | supported |
|---|---|---|---|---|---|
| E2 verb_final (48 items, 2 templates) | 0.41 | 0.17 | +0.24 | [0.11, 0.36] | (apparent) |
| **E6 verb_final (240 items, 8 templates)** | **0.567** | **0.568** | **−0.001** | **[−0.075, 0.071]** | **NO** |

The CI *tightened* (width 0.145 vs 0.255) and the effect **vanished**.

## Root cause (diagnosed)
Two defects in the original E1 `verb_final` subset:
1. **Only 2 templates** → `GroupKFold(4)` is degenerate (cannot form 4 valid folds from 2 groups);
   the exploratory per-structure CV for verb_final was not sound.
2. **Context→label correlation**: with 2 templates and fixed objects ("harbor"/"report"), each
   verb class co-occurred with a narrow context, so a *text* probe could decode the verb class
   **early** from context words → artificially LOW text holism → apparent speech advantage.
When the verb is genuinely unpredictable until the last word (E6's full crossing), **both** speech
and text show high holism (~0.57) and there is no difference.

## Consequences (stated plainly)
- The **E2b/E3 "replications" replicated the same confounded 48-item subset**, so they do not
  rescue the claim — they inherit the artifact.
- The E2 **primary** test P2 (speech>text on the pooled `late` set) drew its signal from these same
  verb_final items (garden_path contributed a clean null); so **P2 is also downgraded** — its
  +0.093 was carried by the artifact, not a robust speech-vs-text difference.
- **Net**: the headline "self-supervised speech encoders resolve meaning more holistically than
  text on late-resolving sentences" is **NOT supported** once stimuli are properly crossed and CV
  is valid. The honest current claim is the null: on well-controlled verb-final speech, holism is
  high in both modalities and does not differ.

## Why this is a good outcome
This is the autoresearch loop and the pre-registration discipline working as intended: an
under-powered exploratory signal was promoted to a confirmatory test, and a *scaling* replication —
proposed by the EFE selector itself — falsified it. Honest negatives are shipped results. The
project's headline is corrected here rather than propagated.

## What survives
- E5's definitional prosody gap (speech recovers a prosodic contrast text structurally cannot).
- The *methodology* (pre-registration, grouped CV, dual gates, the loop) — which is exactly what
  caught the error.
- The digital-edition pillar (M0–M3) is untouched by this.

## Definitive re-run (E7) — question settled
Pre-registered (`research/prereg/H-HOLISM-E7.md`), matched vocabulary (same 8 verbs in early vs
late position, 240+240 items, one grouped-CV probe across both, full power):
- **P1** HI(late) > HI(early): strongly supported in **both** modalities (speech +0.435
  CI [0.379, 0.489]; text +0.576 CI [0.521, 0.630]). The metric is valid — genuinely late-resolving
  meaning yields high holism; early meaning near-zero.
- **P2** HI_speech(late) − HI_text(late): **null, slightly reversed** — −0.075, CI [−0.148,
  −0.0004], p=0.98 in the predicted direction. No speech advantage; if anything text is marginally
  more holistic.

**Verdict (final):** there is **no speech-vs-text holism difference** in this decodability-trajectory
paradigm once stimuli are properly crossed and CV is valid. The E2 headline is fully retired. What
stands is a careful **null** plus a working paradigm (P1) — and a documented case of a false
positive caught and corrected by pre-registration + scaling + the autoresearch loop.
