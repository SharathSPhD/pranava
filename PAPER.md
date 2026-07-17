# Pranava: A Digital Vākyapadīya and an Honest Test of Speech-Centred Cognition in Language Models

*Working paper — Pranava project, 2026-07. All results reproducible via `gates/check.py`.*

## Abstract

The claim that machine cognition should begin from continuous sound rather than discrete text —
grounded in Bhartṛhari's *śabdādvaita*, the neuroscience of inner speech, and the frontier of
speech language models — is provocative but rarely tested. We do two things. First, we build a
complete, tested, verse-anchored **digital critical edition of the *Vākyapadīya*** (1,797 kārikās
with translation, commentary, IAST, morphology, and a concept graph). Second, we **empirically
test** the strongest computational reading of the sphoṭa/*pratibhā* thesis — that self-supervised
speech encoders resolve sentence-meaning more *holistically* (as a late "flash") than text models.
An initial result appeared to confirm this and replicated across four model pairings; a
pre-registered **scaling replication, proposed by our own autoresearch loop, falsified it**, and a
matched-vocabulary re-run settled the question: **there is no speech-vs-text holism difference**
once stimuli are properly controlled. We report the false-positive-and-correction in full as a
methodological result. What survives is a measured sense in which speech carries pragmatic
(prosodic) information text structurally cannot, a working Navya-Nyāya **epistemic gate** that
independently reproduces our empirical verdict, and a grounded reference spec for the
Neuro-Symbolic Acoustic Model that labels each claim evidenced / analogical / open.

## 1. Motivation

Text LLMs operate on already-segmented symbols. Three traditions converge on the idea that
cognition instead begins as a continuous acoustic field that bursts into discrete meaning: (i)
Bhartṛhari's *Vākyapadīya*, where the continuous *dhvani* is segmented by the flash of meaning
(*sphoṭa*) and sentence-meaning is a distinct intuition (*pratibhā*, VP 2.143); (ii) the triple-
network neuroscience of inner speech; (iii) speech LMs (Mimi/Moshi, PAST, DyCAST) moving toward
continuous/semantic-acoustic representations. Pranava asks whether the computational reading of
this convergence is *true*, not merely appealing.

## 2. Pillar I — the digital edition (M0–M3, M2b)

- **M0**: 1,797 mūla kārikās aligned with a complete original translation + commentary; 286 verses
  flagged *contested*; 100% coverage; no fabricated translations.
- **M1**: IAST for every line; Devanāgarī integrity verified.
- **M2/M2b**: morphology via Saṃsādhanī (authoritative); the container's segmenter is broken
  (Heritage backend absent), so we add vidyut-cheda splitting **validated** by Saṃsādhanī —
  coverage rises 0.455 → 0.533. The ≥0.85 aspiration is not met with off-the-shelf tools; we say so.
- **M3**: a concept knowledge graph — 20 technical terms empirically anchored to verses (quoted
  mūla), 125 co-occurrence edges, 6 hand-verified doctrinal relations (e.g. *śabda*=*brahman* @1.1;
  *vākya* grasped-by *pratibhā* @2.143, the locus classicus).

This pillar is a reusable scholarly asset independent of the experimental findings.

## 3. Pillar II — the holism investigation and its correction

### 3.1 Method
288 controlled utterances (early- vs late-resolving; garden-path, verb-final, verb-first),
single-voice TTS. For a speech model (WavLM) and a text model (GPT-2), a template-grouped-CV linear
probe reads meaning decodability at each relative position; the **Holism Index (HI)** = fraction of
decodability gain arriving in the last 20% (high = late flash). Pre-registered before results.

### 3.2 The apparent finding (E2, E2b, E3)
On the verb-final subset, speech HI (0.41) exceeded text HI (0.17); the gap replicated across
WavLM/HuBERT × GPT-2/distilgpt2 — four pairings, all CIs excluding zero. This looked robust.

### 3.3 The correction (E6, E7)
Our autoresearch loop (§4) proposed *scaling* the verb-final set to tighten the CI. With a properly
crossed 240-item design (8 templates × 6 objects × 8 verbs, valid grouped CV), **the effect
vanished** (−0.001, CI [−0.075, 0.071]). Diagnosis: the original subset had only 2 templates
(degenerate grouped-CV) and context→verb correlations that let the *text* probe decode early,
faking low text-holism. The E2b/E3 "replications" had reused that confounded subset. A definitive
matched-vocabulary re-run (E7; same 8 verbs early vs late, full power) confirmed: **P1** (late>early
holism) holds strongly in *both* modalities — the metric is valid — while **P2** (speech>text) is
null and slightly reversed (−0.07, CI [−0.148, −0.0004]). **Net: no speech-vs-text holism
difference.**

### 3.4 What does hold (E5)
Speech carries prosodic information text cannot: on a controlled prosodic contrast, a probe decodes
prosody from speech at ceiling and from (identical) text at exact chance — a definitional gap of
0.5. An exploratory attempt to *localise* prosody by layer failed (the manipulation saturates all
layers) — reported as a negative.

## 4. Method as contribution: autoresearch loop + pre-registration (X0)

We reuse prabodha's Expected-Free-Energy selector to propose the next experiment by information
gain per GPU-hour, with our dual-verdict gates as tiered observations and a re-entrant ledger. It
drove the E3/E5/E6 cycles — and it was the loop's own "scale to tighten the CI" proposal that
surfaced the false positive. Combined with pre-registration (committed before every experiment;
git proves ordering) and template-grouped CV, this pipeline **caught and corrected** an error that
four naïve replications had entrenched. This is the paper's central methodological claim: the
discipline mattered more than any single result.

## 5. Epistemic layer (X1) and the NSM spec (X2)

We reuse the operator's published *pramana* (arXiv 2604.04937) Navya-Nyāya models to build an
epistemic auditor: a claim is *ascertained* only if it clears saṃśaya → pramāṇa → pañca-avayava →
tarka → hetvābhāsa → nirṇaya. Run over our own claims, it **independently refuses** the retracted
holism claim as *savyabhichāra* (inconclusive reason) + *satpratipakṣa* (counter-balanced) — the
darśana framework reproducing the empirical verdict — while ascertaining the textual claims and the
corrected null. The NSM reference spec (X2) then specifies the four-layer architecture with each
claim labelled evidenced / analogical / open: Layer A (acoustic continuum) is *evidenced
non-redundant*; Layer B (sphoṭa extraction) has a real late-emergence phenomenon but *no
speech-specific advantage*; Layer C (pramāṇa) is *built*; Layer D (workspace) is *analogical*.

## 6. Limitations

Synthetic single-speaker TTS; small closed vocabulary; a specific decodability-trajectory
operationalisation of "holism"; WavLM/GPT-2-scale models; the prosody result is definitional. The
holism null is specific to this paradigm — an online/causal decoding design (Layer B, open) might
behave differently. The edition's morphology is capped by off-the-shelf segmentation quality.

## 7. Conclusion

Pranava delivers a reusable digital *Vākyapadīya*; a measured pragmatic-information gap favouring
speech; a carefully established **null** on speech-vs-text holism with a documented self-correction;
a working autoresearch loop and Navya-Nyāya epistemic gate; and an honest NSM spec. The most
valuable outcome is not a confirmation but a correction — evidence that the pre-registration +
grouped-CV + autoresearch discipline can catch a compelling false positive before it propagates.
The sphoṭa thesis is not thereby refuted; its naïve computational reading is, and the ground is now
cleared for a better one.

## Reproducibility

`uv pip install -e ".[dev]"`; `pytest` (85 tests); `python gates/check.py` (13 milestones, dual
gates). Pre-registrations in `research/prereg/`; per-experiment reports in `research/`; the
NSM spec in `specs/NSM-reference-spec.md`.
