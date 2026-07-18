# Pre-registration — P3 battery: steering uptake, nyāya guardrail, fusion-v2
Registered 2026-07-18, BEFORE any battery run. Models: the XL-retrained 200M (xl_ckpt.pt) and, when
available, the 1.13B XL (xl1b_ckpt.pt). Frozen 58-clip val throughout. Tiers per prabodha RULES R6
(smoke → screen → confirm); claims only at the tier actually run.

## S-U — Sphoṭa-workspace steering uptake (prabodha port)
**Question.** Does writing a concept direction into the ALM's sphoṭa workspace band produce
readback-verified uptake (prabodha criteria) beyond a matched random-direction control?
**Method.** Re-fit the sphoṭa-lens band on the eval model (bands may shift post-retrain). For each of
≥30 val clips: concept = a gold kāraka filler's byte sequence (target word). Inject
α·‖h‖·direction at the band's layers during greedy decode (existing steer.greedy_steered), α ∈
{0.05, 0.1, 0.2, 0.4}. Readback: rank of the concept's first byte in the logits at t+1..t+3;
entropy before/after.
**Controls.** (a) random unit direction, same α·‖h‖ norm; (b) α=0.
**Pre-registered outcomes.**
- H-SU1 (loaded): concept rank enters top-5 at ≥2× the random-control rate at some α.
- H-SU2 (behavioral): steered output contains the concept word at ≥2× the control rate.
- H-SU3 (svātantrya budget): |entropy delta| ≤ 0.5 nats at the α that satisfies H-SU1 — steering
  that only works by collapsing the distribution FAILS.
- Mala classification (āṇava/māyīya/kārma/svātantrya) reported for every failure.
**Falsifiable:** if uptake ≈ control at every α within budget, steering does not work on this
model — report as an honest negative.

## N-G — Nyāya guardrail at decode time (śabda-pramāṇa self-consistency)
**Question.** Does a nyāya-style legality constraint — the model's kāraka answer must be grounded in
its OWN heard transcription (anumāna from śabda; NO gold access) — improve kāraka accuracy?
**Method.** For each val clip and task ∈ {karta, karma, kriya, karana}: (1) decode the transcription
(free, EOS); (2) decode the kāraka answer (free, EOS); (3) legality check: answer must match a word
of the transcription (edit distance ≤ 1); (4) if illegal → constrained re-decode: byte-trie over
transcription words masks illegal continuations (greedy over legal byte prefixes only).
**Pre-registered outcomes.**
- H-NG1: guardrailed kāraka exact-match ≥ unguarded (strictly greater to claim benefit).
- H-NG2: fire-rate and fix/break/neutral counts reported; a guardrail that fires often but fixes
  nothing is reported as such.
- Zero gold leakage by construction (constraint set derives only from the model's own transcript).
**Falsifiable:** if constrained re-decode reduces accuracy (the transcript itself too noisy), that
is the honest result and bounds the approach by transcription quality.

## F-2 — Fusion-v2: causal cross-modal integration profile
**Question.** WHERE does audio information become causally necessary for the text-side meaning —
i.e. the sphoṭa locus measured causally, not by CKA similarity (v1's admitted weakness)?
**Method.** For each layer ℓ: run the model with audio-position hidden states at ℓ replaced by their
batch-mean (information cut at ℓ); measure kriyā-decodability (the emergence.py linear probe) from
text positions at the final layer. Integration(ℓ) = decodability(intact) − decodability(cut at ℓ).
**Pre-registered outcomes.**
- H-F2a: Integration(ℓ) has an interior peak (not layer 0/L) — a genuine fusion locus.
- H-F2b: the causal peak lies within ±2 layers of the correlational sphoṭa layer (13 on the old
  model; re-measured on the eval model) — corroboration; if not, the discrepancy is reported and the
  causal number wins.
**Chance level** from label permutation (100 shuffles), as in emergence.py.

## Metrics discipline
Every script writes phase-specific artifacts (p3su_*, p3ng_*, p3f2_*) — no clobbering. Gates SU/NG/F2
assert methodology + honest reporting, not predetermined winners.
