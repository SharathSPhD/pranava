# NSM Layer C — a gold-free pramāṇa gate that predicts ALM correctness

Layer C sits after generation and audits each ALM utterance through a Navya-Nyāya lens (reusing the
operator's `pramana` auditor). It asks: is this output *ascertained* (nirṇaya), or a fallacy-tainted
guess? Crucially it uses **no gold label** — only three observable properties of the byte-decoder's
output:

- **pratyakṣa** — decode confidence (mean per-token max-softmax probability);
- **śabda** — romanization legality (a real utterance uses legal script, not stray bytes);
- **anumāna** — non-repetition (looping infers an invalid generation).

Confidence is the primary support; legality and repetition are *disqualifiers* that only ever count
against a claim (so a low-confidence output can't be inflated to ascertained by being merely legal).

## The test, and an honest first miss
The question a confidence filter must answer: **do the outputs it accepts turn out to be more correct
than the ones it rejects?** The first attempt tested this on `transcribe` and **failed** — CER 0.947
(ascertained) vs 0.934 (rejected), essentially no separation. The reason was diagnosable, not a bug:
the instruction model's `transcribe` is *uniformly poor* (~0.94 CER, and the set mixes English the
Sanskrit core cannot transcribe at all), so there is **no correctness variance for any filter to
detect**. A confidence gate can only be validated where the model is sometimes right and sometimes
wrong.

## The result (held-out, 273 instruction examples with real right/wrong variance)
Re-run on the extractive kāraka + language tasks, scored by exact-match:

| bucket | n | accuracy |
|---|---|---|
| **ascertained** | 264 | **0.398** |
| **rejected** | 9 | **0.000** |

**Layer C's rejections are 100% precise: every output it flagged as not-ascertained was in fact
wrong** (0 of 9 correct), while ascertained outputs are ~40% correct — an accuracy gap of 0.398,
achieved with no gold label. Gated: `python gates/check.py NC`.

## Honest reading — precise, not exhaustive
Layer C is a **high-precision, low-recall** filter: it is conservative, flagging only 9 of the ~165
wrong answers (those with low confidence or repetition), but everything it flags is genuinely wrong.
It does *not* catch confidently-wrong outputs — the model is sometimes calibratedly certain of a
mistake, and a gold-free gate cannot see that. So Layer C is a trustworthy *reject* signal (if it
says "not ascertained," don't trust the output), not a complete error detector. That is exactly the
epistemic role the NSM design asks of it — refuse ascertainment when the evidence is weak — and it
is the confidence channel the platform's API and app can surface. Artifacts:
`data/alm/layer_c_validation.json`, `data/alm/layer_c_records.json`; see [`NSM.md`](../NSM.md) for
where Layer C sits in the full stack.
