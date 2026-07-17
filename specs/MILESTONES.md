# Pranava — Objective Milestone Ledger

Every milestone has a **gate**: a machine-checkable predicate (a test, a script exit code, a metric
threshold, or an artifact-on-disk check). A milestone is `DONE` only when its gate passes in CI
(`gates/check.py`). No milestone is marked done by narrative. Negative empirical results still pass
their gate if the experiment ran validly and is honestly reported.

Legend: ☑ done · ◐ in progress · ☐ not started

---

## Pillar I — ŚABDA (digital critical edition + KG)

### M0 — Aligned edition builds & is fully tested  ☑ DONE (2026-07-16)
- **Gate**: `pytest tests/corpus` green; `scripts/build_edition.py` emits
  `data/vakyapadiya/edition.jsonl` with 1797 kārikās and `coverage.json` with
  `translated_fraction == 1.0`, `contested == 286`.
- **Evidence**: 16 tests pass; edition.jsonl (1797 lines) + coverage.json on disk.

### M1 — IAST transliteration + integrity gate  ☑ DONE (2026-07-16)
- Devanāgarī→IAST for every mūla line; daṇḍa/verse-number decoration stripped, not mangled.
- **Gate**: `gates/check.py M1` — `test_translit.py` green (golden pairs incl. avagraha; daṇḍa
  hygiene; full-edition integrity: all ~3,594 lines ≥0.5 Devanāgarī ratio and transliterate);
  edition.jsonl carries `iast_lines` for all 1797 rows, none missing.
- **Evidence**: 9 tests pass; edition.jsonl regenerated with IAST; M1 gate PASS.

### M2 — Morphological analysis via Saṃsādhanī  ☑ DONE (2026-07-16)  ·  rescoped (see below)
- Wired the running `samsaadhanii` morph.cgi (localhost:8090, JSON mode) via a tested client;
  ran real morphological analysis over the whole Brahma-kāṇḍa; persisted every analysis.
- **Honest rescope**: the container's sandhi segmenter + dependency parser are **non-functional**
  (Heritage `sktgraph2` backend absent — `research/01-samsaadhanii-integration.md`). Full
  kāraka/dependency parse is therefore moved to **M2b (blocked)**. No parse was fabricated.
- **Gate** (`gates/check.py M2`, dual): code = morph client tests green; domain = ran over
  144 kāṇḍa-1 verses / 1189 whitespace tokens, **coverage 0.455** (541 tokens with ≥1 valid
  analysis; the rest are unsplit sandhi/compounds). Number reported, not asserted.
- **Evidence**: `data/vakyapadiya/morph_kanda1.jsonl` (per-token analyses) + report; gate_M2.json.

### M2b — Full dependency/kāraka parse (segmentation)  ☐ BLOCKED
- Needs a working segmenter. Candidate: `vidyut-cheda` (pip `vidyut` installed; needs data kosha).
  Whitespace coverage of 0.455 shows segmentation is the main lever to lift word-level coverage.
- **Gate**: with segmentation, ≥0.85 token coverage on Brahma-kāṇḍa; a 20-verse gold set
  spot-checked and documented.

### M3 — Concept knowledge graph  ☑ DONE (2026-07-16)
- 20 core concepts, each **empirically anchored** to verses whose mūla genuinely contains the
  stem (diacritic-folded IAST match), quoting the original line; 125 verse-grounded co-occurrence
  edges; 6 curated doctrinal relations, each verse-anchored and hand-verified
  (e.g. `śabda identified-with brahma @1.1`; `vākya grasped-by pratibhā @2.143` — the locus
  classicus; `sphoṭa manifested-by dhvani @1.75`).
- **Gate** (`gates/check.py M3`, dual): code = 7 KG tests green (incl. "anchors actually contain
  the term", "edges grounded", "curated relations verse-anchored"); domain = ≥12 concepts all
  anchored, edges>0, ≥4 curated relations all anchored.
- **Evidence**: `data/vakyapadiya/concept_graph.json` (71 KB); gate_M3.json.

### M4 — Published, queryable edition  ☐
- Static site + JSON API (Vercel) and/or Supabase Postgres with the edition, parses, and KG.
- **Gate**: deployed URL returns verse 1.1 with mūla+IAST+translation+concept links; a smoke
  test hits the live endpoint; data provenance & license documented.

---

## Pillar II — SPHOṬA-BENCH (empirical study)

### E0 — Speech-model harness on GB10  ☑ DONE (2026-07-16)
- `pranava.speech.harness.SpeechEncoder` loads WavLM-base on CUDA and extracts per-frame,
  per-layer hidden states `[13, n_frames, 768]` at ~50 Hz. Whisper GPU transcription also
  validated (research/transcripts/).
- **Gate** (`gates/check.py E0`, dual): code = 5 harness tests green on GPU (device, shapes,
  determinism, frame-rate, resample); domain = real clip → 299 frames @ 49.83 Hz on `cuda:0`
  in 0.42 s, manifest + last-layer tensor persisted.
- **Evidence**: `data/speech/e0_manifest.json`, `e0_lastlayer.npy`; gate_E0.json.

### E1 — Stimulus set with ground truth  ☑ DONE (2026-07-16)
- 288 controlled items, balanced early(168)/late(120) across 4 structures (canonical, garden_path,
  verb_final, verb_first), 11 meaning labels, exact disambiguation indices by construction.
  Synthesized with a single fixed neural voice (piper en_US-lessac-medium, 22.05 kHz, 483 s total).
  Every item carries a text-parallel condition (the written sentence). Datasheet documents limits.
- **Gate** (`gates/check.py E1`, dual): code = 9 stimulus tests green (size, balance, late-items-
  disambiguate-late, determinism); domain = 288 items, 288 WAVs present, all labelled, both
  resolution classes.
- **Evidence**: `data/stimuli/manifest.jsonl`, `datasheet.json`, 288 WAVs (local); gate_E1.json.

### E2 — Hypothesis H-HOLISM (pre-registered)  ☑ DONE (2026-07-16)
- Ran the pre-registered experiment: Holism Index from a template-grouped-CV linear probe over
  WavLM (speech) vs GPT-2 (text) decodability trajectories, 3 seeds, bootstrap + Holm.
- **Result (honest, nuanced)**: **P2 supported** — speech resolves *late-resolving* sentences more
  holistically than text (HI +0.093, CI [0.016, 0.167], p=0.0095, Holm). Confound checked:
  garden-path items (structurally cued) show **no** effect (0.128 vs 0.131); the signal is carried
  by verb-final items where meaning genuinely resolves last (HI 0.41 vs 0.17, CI [0.106, 0.361]).
  P1 (within-modality late>early) not supported. Full reading + threats in `research/E2-report.md`.
- **Gate** (`gates/check.py E2`, dual, validity-not-outcome): code = 18 experiment/metric tests
  green; domain = pre-reg + report + figure present, probe beats chance (0.47/0.59 vs 0.091),
  ≥3 confirmatory tests each with CI + bootstrap p.
- **Evidence**: `data/experiments/e2_results.json`, `e2_trajectories.png`; gate_E2.json.

### E3 — Second causal text model (loop-driven)  ☑ DONE (2026-07-16, via X0)
- Reused E2 pipeline with distilgpt2; the holism finding replicated (speech verb_final HI 0.406 vs
  0.148, CI [0.128, 0.384], p=0.0005). See X0.

### E5 — H-PROSODY: the acoustic-blindness gap (pre-registered, loop-proposed)  ☑ DONE (2026-07-16)
- Quantified the speech-vs-text pragmatic gap on a controlled prosodic contrast: speech decodes
  prosody (acc 1.0), text is exactly at chance (0.5) — **gap 0.5, CI [0.425, 0.581]**.
- **Honest negative**: the pre-registered *exploratory* layer-localization hypothesis is NOT
  supported — the rate+pitch manipulation saturates all 13 layers, so prosody depth cannot be
  compared to meaning depth. Recorded to the loop ledger as a near-miss (tier 1) so the selector
  refines rather than repeats. Full reading: `research/E5-report.md`.
- **Gate** (`gates/check.py E5`, validity-not-outcome): pre-reg + report present, PR1/PR2 recorded,
  gap has CI. Evidence: `data/experiments/e5_results.json`, gate_E5.json.

### E3-orig — Hypothesis H-VERBFIRST (pre-registered)  ☐ (superseded by E2 exploratory + E3/E2b)
- The verb-first/event-first question is partly answered by E2's structure breakdown (verb_first
  items: speech HI 0.377 vs text 0.022). A dedicated pre-registered version remains optional.

### E6 — Verb-final scaling replication → CORRECTS E2  ☑ DONE (2026-07-17)
- Loop-proposed ("scale to tighten the CI"). Built a properly-crossed 240-item verb-final pool
  (8 templates × 6 objects × 8 verbs; valid grouped CV) and re-ran the speech-vs-text holism test.
- **Corrective negative**: the effect **vanishes** — HI speech 0.567 ≈ text 0.568, effect −0.001,
  CI [−0.075, 0.071] (CI tightened to 0.145 from E2's 0.255). Root cause: E1's verb_final subset
  had only 2 templates (degenerate GroupKFold) + context→verb correlation letting text decode
  early. **E2/E2b/E3's "4-way replication" reused that confounded subset → the headline holism
  claim is NOT supported.** Full walkback: `research/E6-correction.md`; E2-report + README corrected.
- **Gate** (`gates/check.py E6`, validity-not-outcome): a rigorous corrective negative is a
  successful milestone — n≥200, valid CV, CI tightened, correction doc present.
- **Evidence**: `data/experiments/e6_results.json`, gate_E6.json.

### E7 — Definitive matched-vocabulary holism test  ☑ DONE (2026-07-17)
- Pre-registered. Same 8 verbs in early vs late position (240+240 items, one grouped-CV probe,
  full power) to settle the question after E6.
- **Verdict**: **P1** (late>early holism) strongly holds in both modalities (speech +0.435, text
  +0.576 — the metric is valid); **P2** (speech>text on late) is **null, slightly reversed**
  (−0.075, CI [−0.148, −0.0004], p=0.98). **No speech-vs-text holism difference exists.** The E2
  headline is fully and definitively retired.
- **Gate** (`gates/check.py E7`): matched pools ≥200, P1 sanity holds, P2 reported with CI, verdict
  recorded. Evidence: `data/experiments/e7_results.json`; gate_E7.json; `research/E6-correction.md`.

### E4 — Report  ☐
- Write up Pillar II as a short paper (methods, results incl. negatives, threats to validity).
- **Gate**: every figure regenerated by a script from committed data; claims traceable to a gate.

---

## Cross-cutting

### X0 — Autoresearch loop wired  ☑ DONE (2026-07-16)
- **Reuses** (not copies) prabodha's `EFESelector` (imported via a venv `.pth`) to drive
  Sphoṭa-Bench: a Candidate menu of next experiments, `gate_*.json` → tiered `Observation`, a
  JSONL ledger (`research/efe_ledger.jsonl`) making beliefs re-entrant. Explore→confirm emerges
  (cheap 'smoke' actions first).
- **Closed a full cycle live**: seeded from gate_E2 → proposed `e3_second_text_model` → agent
  *disposed* (BERT is bidirectional → substituted causal distilgpt2, documented) → ran E3 →
  observed gate_E3 (tier 3) → recorded → loop re-proposed `e5_prosody_manipulation`.
- **E3 result**: the holism finding replicates on a *third* model (distilgpt2: speech verb_final
  HI 0.406 vs 0.148, effect +0.258, CI [0.128, 0.384], p=0.0005). Now robust across 2 speech
  encoders (WavLM, HuBERT) × 2 causal text models (GPT-2, distilgpt2).
- **Gate** (`gates/check.py X0`, dual): code = 9 loop tests green; domain = ledger shows a full
  propose→observe→re-propose cycle (obs=2, prop=2).
- **Evidence**: `research/efe_ledger.jsonl`, `data/experiments/e3_results.json`, gate_X0/E3.json.

### X1 — Pramāṇa validation layer (reuse)  ☐
- Reuse the operator's published Navya-Nyāya reasoning (pramana) as the "Layer C" validator over
  extracted claims in the KG / experiment conclusions.
- **Gate**: at least one claim passed through the 6-phase epistemic check with recorded output.

### X2 — NSM reference spec  ☐
- A grounded spec of the Neuro-Symbolic Acoustic Model, citing *only* what M/E milestones
  established, with each layer's assumptions labelled evidenced / analogical / open.
- **Gate**: spec review checklist; no claim without a milestone or citation backing it.
