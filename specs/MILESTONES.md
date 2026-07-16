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

### E0 — Speech-model harness on GB10  ☐
- Load an open speech LM / SSL encoder (Whisper working ☑; add a semantic-token model:
  Mimi/HuBERT/WavLM) and extract per-frame hidden states on GPU.
- **Gate**: a script produces hidden-state tensors for a fixed audio set, shapes asserted,
  runs on CUDA (not CPU fallback), timing logged.

### E1 — Stimulus set with ground truth  ☐
- A controlled spoken stimulus set (minimal pairs; sentences where meaning resolves late;
  verb-first vs noun-first variants; prosody-carrying sarcasm/urgency items). TTS-generated +
  provenance; text-parallel condition for every audio item.
- **Gate**: ≥200 items, each with structured labels; a datasheet documents generation & limits;
  held-out split defined before any modeling.

### E2 — Hypothesis H-HOLISM (pre-registered)  ☐
- Probe decodability of final-sentence-meaning across time; test whether the curve shows a late
  near-discontinuous jump (sphoṭa-like) vs monotone accrual.
- **Gate**: pre-registration committed *before* results (`research/prereg/H-HOLISM.md`); metric,
  null, and stopping rule fixed; ≥3 seeds; bootstrap CI; result (either sign) written up with the
  analysis script reproducing the figure from raw tensors.

### E3 — Hypothesis H-VERBFIRST (pre-registered)  ☐
- Compare emergence/robustness of event vs entity representations in speech vs text models.
- **Gate**: same rigor bar as E2; explicit text-vs-speech contrast; reproducible.

### E4 — Report  ☐
- Write up Pillar II as a short paper (methods, results incl. negatives, threats to validity).
- **Gate**: every figure regenerated by a script from committed data; claims traceable to a gate.

---

## Cross-cutting

### X0 — Autoresearch loop wired  ☐
- Adapt prabodha's loop to drive E2/E3: hypothesis → experiment → verify-gate → refine, logging
  each iteration.
- **Gate**: one full loop iteration recorded end-to-end with artifacts per stage.

### X1 — Pramāṇa validation layer (reuse)  ☐
- Reuse the operator's published Navya-Nyāya reasoning (pramana) as the "Layer C" validator over
  extracted claims in the KG / experiment conclusions.
- **Gate**: at least one claim passed through the 6-phase epistemic check with recorded output.

### X2 — NSM reference spec  ☐
- A grounded spec of the Neuro-Symbolic Acoustic Model, citing *only* what M/E milestones
  established, with each layer's assumptions labelled evidenced / analogical / open.
- **Gate**: spec review checklist; no claim without a milestone or citation backing it.
