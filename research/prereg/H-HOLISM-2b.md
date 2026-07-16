# Pre-registration — H-HOLISM-2b (replication of the verb-final holism finding)

Committed before E2b results. Tests whether E2's central signal is WavLM-specific or a general
property of self-supervised speech encoders.

## Background
E2 found (exploratory) that on **verb_final** items — meaning genuinely resolves at the last word —
the speech model (WavLM-base) concentrates meaning-decodability late (HI 0.41) far more than text
(GPT-2, HI 0.17): effect +0.237, CI [0.106, 0.361]. Garden-path items (structurally cued) showed
no effect, ruling out that confound.

## Hypothesis (directional, confirmatory for this replication)
- **R1**: With a *different* SSL speech encoder (**HuBERT-base**, facebook/hubert-base-ls960),
  the same contrast holds: HI_hubert(verb_final) > HI_text(verb_final). Null: equal.
- **R2**: HuBERT reproduces the speech>text late-holism on late-resolving items overall
  (HI_hubert(late) > HI_text(late)). Null: equal.

## Method
Identical to E2 (same 288 stimuli, layer-9 frames, cumulative mean-pool, template-grouped-CV
linear probe, 3 seeds, Holism Index, bootstrap 2000 + one-sided p). Only the speech model changes.
Text side reuses E2's GPT-2 features. Holm across the 2 tests.

## Stopping / integrity
Fixed dataset. No item exclusion except encode failures (logged). Null is a shipped result: if
HuBERT does NOT replicate, the E2 finding is downgraded to WavLM-specific and reported as such.

## Threats
Same as E2. Additionally: HuBERT and WavLM share training-data lineage (LibriSpeech-family), so
replication shows robustness across architectures/objectives but not across speech domains.
