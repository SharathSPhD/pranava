# NSM — Neuro-Symbolic Acoustic Model: reference spec (X2)

A grounded specification of the architecture the `/docs` material points to — a
**Neuro-Symbolic Acoustic Model** — written *after* Pranava's experiments, so every claim is
labelled by what this project actually established:

- **[EVIDENCED]** — supported by a gated Pranava milestone (with the milestone id).
- **[ANALOGICAL]** — a principled mapping from darśana/neuroscience to architecture, not (yet) tested here.
- **[OPEN]** — a design question Pranava did not resolve (often *because* an experiment nulled a naïve version).

The spec is deliberately conservative: the headline empirical claim that motivated the NSM
(speech is "more holistic" than text) was **falsified** under proper controls (E6/E7), and that
correction is built into the layer descriptions below rather than papered over.

---

## The four layers (from the docs' blueprint)

### Layer A — Acoustic continuum (persistent pre-segmental memory)
Maintain a continuous acoustic representation, not a stream of pre-chunked symbols.
- **[EVIDENCED, E0]** Self-supervised speech encoders (WavLM/HuBERT) provide a per-frame continuous
  representation at ~50 Hz on the GB10; this is a usable Layer-A substrate.
- **[EVIDENCED, E5]** This substrate carries **prosodic/pragmatic** information that a text stream
  structurally cannot (a controlled prosodic contrast: speech-decodable at ceiling, text at exact
  chance — gap 0.5). So Layer A is not redundant with a text front-end: it holds information text loses.
- **[OPEN]** Whether a *dynamic, meaning-driven* segmentation of this continuum (à la DyCAST,
  arXiv 2601.23174) outperforms fixed-frame tokenization for downstream reasoning — not tested here.

### Layer B — Sphoṭa extraction (continuum → discrete meaning)
Isolate meaning-bearing units from the continuum — the "flash" (sphoṭa) of meaning.
- **[EVIDENCED, E2/E7]** Meaning *is* decodable from Layer-A frames, and its decodability rises
  sharply for genuinely late-resolving sentences (Holism Index high for verb-final, low for
  early-resolving — P1 in E7, both modalities). So a late, concentrated emergence of decodable
  meaning is a **real, measurable phenomenon** in the representation — the closest empirical analogue
  Pranava found to the sphoṭa "flash".
- **[CORRECTED / OPEN, E6/E7]** The stronger claim — that *speech* extracts meaning *more
  holistically than text* — did **not** survive proper controls (matched-vocab, powered, valid CV:
  effect −0.07, CI includes 0). So Layer B's "flash" is **not** a speech-specific advantage over
  text under this paradigm. Whether a different operationalization (online/causal decoding,
  predictive-coding surprise signals) reveals a speech-specific effect is **[OPEN]**.
- **[ANALOGICAL]** Bhartṛhari's *pratibhā* (VP 2.143, [EVIDENCED as a textual claim, M3]) — the
  sentence-meaning as a distinct intuition, not the sum of word-meanings — motivates modelling
  Layer B as an emergent whole-utterance readout rather than a token-wise accumulator.

### Layer C — Pramāṇa validation (epistemic gate)
Before a meaning is asserted as knowledge, validate it as *śabda-pramāṇa* — trustworthy testimony —
via Navya-Nyāya's 6-phase check.
- **[EVIDENCED, X1]** Implemented and reused from the operator's published *pramana* (arXiv
  2604.04937): saṃśaya → pramāṇa → pañca-avayava → tarka → hetvābhāsa → nirṇaya, returning an
  ascertainment verdict. Applied self-referentially, it **independently refused** Pranava's own
  retracted holism claim as *savyabhichāra* (inconclusive reason) + *satpratipakṣa* (counter-balanced)
  — matching the empirical correction. This is a working epistemic gate over claims.
- **[ANALOGICAL]** Wiring Layer C *inline* — so the model tags each asserted proposition with its
  pramāṇa type and blocks fallacious ones before output — is specified but not integrated into a
  generation loop here.

### Layer D — Global-workspace integration (jSpace / GNW)
Integrate validated meanings into a verbalizable global workspace (Anthropic j-space / GNW / the
operator's PWM/prabodha line).
- **[ANALOGICAL]** Descends from jSpace and prabodha's recognition-gated workspace steering; Pranava
  did not build Layer D. Recorded as the integration target, not a result.

---

## What Pranava actually established (honest summary)
1. **[M0–M3]** A complete, tested, citable digital critical edition of the *Vākyapadīya* + a
   verse-anchored concept graph — the darśana substrate, machine-usable.
2. **[E5]** A real, quantified sense in which speech representations carry pragmatic information
   text cannot (Layer A is non-redundant).
3. **[E2→E6→E7]** A careful **null**: no speech-vs-text *holism* advantage once stimuli are
   properly controlled — plus a documented false-positive-caught-and-corrected, via
   pre-registration + grouped CV + the autoresearch loop.
4. **[X0]** An autoresearch loop (reusing prabodha's EFE) that genuinely drove the above cycles.
5. **[X1]** A Pramāṇa epistemic gate (reusing pramana) that reproduces the empirical verdict on
   Nyāya grounds.

## What remains genuinely open
- A speech-specific meaning-emergence effect under a causal/online decoding paradigm (Layer B).
- Meaning-driven dynamic segmentation vs fixed frames (Layer A).
- Inline Layer-C validation in a generation loop; Layer-D workspace integration.
- Natural (not synthetic) prosody; multi-speaker; larger/bidirectional text baselines.

## Design stance
The NSM is worth building for Layers A and C, which Pranava *evidenced* as non-trivial and
non-redundant. Layer B should be pursued as an **open research question**, not asserted — the
project's own most important finding is that the naïve version of Layer B's headline claim is false.
This spec exists so the next iteration builds on what is true, not on what was merely hoped.
