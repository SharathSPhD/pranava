# PRANAVA — Vision

> प्रणव · *praṇava* — the primordial sound (Oṃ); the seed-sound from which articulated
> speech and meaning unfold. The name of a project about sound becoming knowledge.

## The one-sentence thesis

**Text language models begin from the wrong abstraction.** They operate on already-segmented
symbols and therefore miss what Bhartṛhari's *Vākyapadīya* (5th c.), the neuroscience of inner
speech, and the current frontier of speech language models all independently point at: cognition
begins as a **continuous acoustic field** that bursts into discrete meaning (*sphoṭa*), and valid
knowledge requires a **normative testimony layer** (*śabda-pramāṇa* / *śābdabodha*) that text LLMs
lack. Pranava builds the scholarly foundation and the falsifiable experiments to take this claim
from essay to evidence.

This is the next movement in a research program the operator has already published
(arXiv 2604.04937, *Pramana*: Navya-Nyāya epistemic reasoning in LLMs) and prototyped across this
machine (prabodha autoresearch loop, PWM, prayoga, jSpace/GNW, panini-data-toolkit, Saṃsādhanī).

## Why now / why this is not a metaphor

Three independent domains converged (see `research/00-landscape-scan.md` and
`research/transcripts/`):

1. **Darśana** — Śabdādvaita: awareness and śabda are coupled; the continuous *dhvani* is
   segmented by the flash of meaning (*sphoṭa*); reality is *verb-first* (action precedes object).
2. **Neuroscience** — inner speech is a triple-network phenomenon (language + sensorimotor + DMN);
   auditory cortex tracks speech at multiple timescales and extracts indivisible units.
3. **AI frontier** — SOTA is shifting from discrete text tokens toward continuous / semantic-acoustic
   representations (Mimi/Moshi, PAST, DyCAST character-aligned dynamic tokenization); benchmarks
   (MMSU, WavBench) expose a real *speech-vs-text semantic gap*.

The recurring blueprint across the operator's own explainers is a **Neuro-Symbolic Acoustic Model
(NSM)**: Layer A continuous acoustic memory → Layer B *sphoṭa* meaning-extraction → Layer C
Navya-Nyāya epistemic validation → global-workspace (jSpace/GNW) integration.

## What "the work" is (scope decision)

Pranava is **not** an attempt to train a frontier speech model from scratch (that would invite
fabricated claims). It is two rigorous, independently-verifiable pillars that together make the
thesis *checkable* and give the NSM blueprint a real foundation:

### Pillar I — ŚABDA: the Vākyapadīya digital critical edition & knowledge graph
The scarce-asset play. Scholarship notes there is *no* comprehensive digital edition of the
Vākyapadīya. We already hold, locally, a complete mūla (1,797 kārikās, Devanāgarī) **and** a
complete original English translation + commentary (1,796 verses, 286 flagged *contested*). Pranava
turns this into a validated, tested, versioned, citable digital edition with:
- canonical verse IDs, mūla↔translation↔commentary alignment (**done, M0**);
- IAST transliteration + morphological/śābdabodha parses via the local Saṃsādhanī service;
- a concept knowledge graph (sphoṭa, dhvani, pratibhā, kāla-śakti, vivarta, …) with verse anchors;
- cross-references to the neuroscience/AI claims, each tagged as *textual* vs *analogical*.
Verifiable by coverage metrics, parse-validity gates, and citation integrity — never by assertion.

### Pillar II — SPHOṬA-BENCH: an empirical test of the continuity/holism thesis
The science. Falsifiable, GPU-run probing of open speech LMs (Mimi/Moshi-class, Whisper, SSL
models) to test predictions the sphoṭa theory makes but text-first models deny — e.g.:
- **Holism**: does meaning resolve as a late, near-discontinuous "flash" (a detectable jump in a
  probe's decodability curve) rather than accreting monotonically token-by-token?
- **Verb-first**: are event/action representations available earlier / more robustly than
  object/noun representations in continuous speech, versus in text?
- **Continuum grounding**: does preserving the pre-segmental acoustic field measurably reduce the
  pragmatic errors (sarcasm/urgency) that text-only pipelines make?
Each hypothesis gets a pre-registered metric, a null, and a gate. Negative results are published.

Both pillars feed a third, longer-horizon deliverable — an **NSM reference spec** grounded in what
Pillars I and II actually establish, plus a Pramāṇa validation layer reusing the operator's
published Navya-Nyāya work.

## Non-negotiables (from the Sākṣī)
No fabrication, no stubs-as-done, no bypassed gates, no invented metrics. Every milestone is backed
by runnable evidence on disk. Sanskrit sources cited accurately; textual claims kept distinct from
computational analogies. Use the GB10 GPU. Build on the incomplete prior work, don't restart it.

## How we work
Spec/PRD-driven; TDD (red→green→refactor); contracts + gates adapted from prabodha; an autoresearch
loop (also from prabodha) driving Pillar II's hypothesis→experiment→verify→iterate cycle; TRIZ and
the attractor-flow / pratyabhijna creative engines for ideation and convergence diagnostics.

## Definition of "phenomenal"
A digital edition good enough that a Sanskritist would cite it and an ML researcher would build on
it; and at least one **falsifiable, GPU-verified** empirical result about speech-vs-text cognition
that is novel, reproducible, and honestly reported — whichever way it comes out.
