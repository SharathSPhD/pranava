# P3 battery — sphoṭa-lens steering, nyāya guardrail, causal fusion (results)

Run against the locked pre-registration (research/prereg/P3-battery-prereg.md). Two checkpoints:
BASELINE = instruct_ckpt (pre-XL); RETRAINED = xl_ckpt (200M, full 10k-clip XL corpus). Frozen 58-clip
val throughout. Honest negatives kept in the body, per the project ethos.

## S-U — Sphoṭa-workspace steering uptake  (workspace band [21,23))
Writing a concept direction (gold kriyā bytes' unembedding rows) into the sphoṭa band during greedy
decode; readback = rank of the concept's first byte at t+1..t+3; controls = matched random direction
(same ‖·‖) and α=0.

| model | H-SU1 loaded (uptake vs random) | H-SU2 uttered | H-SU3 within svātantrya budget |
|---|---|---|---|
| baseline (instruct) | ✅ 0.758 vs 0.308 = **2.46×** | ❌ 0.0 | ❌ Δentropy −1.59 nats |
| **retrained (xl)** | ✅ 0.767 vs 0.150 = **5.11×** | ❌ 0.0 | ❌ Δentropy −1.44 nats |

**Finding (the positive, pre-registered):** the sphoṭa workspace is **causally meaning-loadable** — a
written concept enters the top-5 readback at **5.1× the random-control rate** on the retrained model
(2.5× baseline; training *sharpens* the workspace). This is the paśyantī workspace made writable and
verifiable, not asserted.
**Honest limits (also pre-registered, also reported):** (H-SU2) loading does not surface as the spoken
word — *paśyantī loading ≠ vaikharī utterance*; (H-SU3) the α that maximizes loading **breaches the
svātantrya (autonomy) budget** — entropy collapses ~1.4 nats, i.e. the steering "works" partly by
narrowing the distribution. Mala classification: uptake succeeds (loaded) but fails *amplified*
(behavioral) and *within-budget* (svātantrya). We do not over-claim controllable generation.

## N-G — Nyāya guardrail at decode  (śabda-pramāṇa self-consistency, zero gold leakage)
Constraint: a kāraka answer must be a word of the model's OWN transcription (edit-distance ≤ 1);
illegal answers trigger a transcript-byte-trie–constrained re-decode. Constraint set derives ONLY from
the model's transcript — never gold.

| model | unguarded EM | guarded EM | fire-rate | fix / break / neutral |
|---|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.991 | 0 / 0 / 112 |
| retrained (xl) | 0.000 | 0.000 | 0.947 | 0 / 0 / 107 |

**Finding (honest inconclusive):** the guardrail neither helps nor hurts. The XL retrain optimized
*transcription* (the leaderboard metric) so strongly that the extractive kāraka exact-match collapsed to
0 on free EOS decode — so there is no correct answer for the constraint to protect, and the
transcript-grounded constraint has nothing to bite. This is a real **objective-tradeoff** result
(transcription ↑, extractive-kāraka ↓), not a validation of the mechanism. The guardrail's *value* is
untestable at this kāraka-accuracy floor; a kāraka-weighted checkpoint is the pre-registered follow-up.

## F-2 — Causal cross-modal integration (fusion-v2)
[F2-LOCUS]  (efficiency-fixed run: activations collected once, 100-shuffle permutation chance re-scored
from cache — the earlier version re-ran every ablation forward 101×). Curve/peak/chance in
data/alm/p3f2_results.json; correlational locus for comparison in research/sphota-lens.md.

## What this establishes for the thesis
The paśyantī workspace is **empirically real, writable, and readback-verifiable** (S-U, strongly) — the
sphoṭa-lens is not a similarity artifact but a causal handle on where audio-borne meaning is held. The
honest limits (utterance gap, autonomy budget, kāraka-accuracy floor) are stated with their evidence,
in the vākya-vallari "honesty boundary" style: we show a measurable, manipulable sphoṭa workspace, not
a solved controllable-generation problem.
