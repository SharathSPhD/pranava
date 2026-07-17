# PRANAVA

**Sound becoming knowledge — a computational study of Bhartṛhari's *Vākyapadīya* and
speech-centred cognition in language models.**

> प्रणव · *praṇava* — the primordial sound from which articulated speech and meaning unfold.

Text language models begin from already-segmented symbols. Bhartṛhari's *śabdādvaita* (5th c.),
the neuroscience of inner speech, and the frontier of speech language models all suggest cognition
begins as a **continuous acoustic field** that bursts into discrete meaning (*sphoṭa*), grasped as
a unified intuition (*pratibhā*), and validated by a normative testimony layer (*śabda-pramāṇa*).
Pranava turns that thesis from essay into **tested code, cited scholarship, and falsifiable
experiments**. See [`VISION.md`](VISION.md).

## Two pillars

**I · ŚABDA — the Vākyapadīya digital critical edition + knowledge graph.** A validated, tested,
citable machine edition of all 1,797 kārikās with a complete original translation + commentary,
IAST, morphological analysis, and a verse-anchored concept graph.

**II · SPHOṬA-BENCH — the empirical study.** GPU-run probes testing where the sphoṭa/pratibhā
picture actually holds in speech vs text models. First result (E2) suggested speech resolves late-resolving meaning more holistically than text — but a properly-powered scaling replication (**E6**) *falsified* it (the original stimuli were under-powered and confounded). Net current claim: **no demonstrated speech-vs-text holism difference**. This self-correction, caught by the pre-registration + autoresearch discipline, is the honest headline — see [`research/E6-correction.md`](research/E6-correction.md).

## Status — every milestone is machine-gated

| Milestone | What | Gate |
|---|---|---|
| M0 | Aligned edition (1,797 kārikās, 100% translation, 286 contested) | ✅ |
| M1 | IAST transliteration + Devanāgarī integrity | ✅ |
| M2 | Morphological analysis via Saṃsādhanī (0.455 coverage; segmenter absent → M2b) | ✅ |
| M3 | Verse-anchored concept knowledge graph (20 concepts, 125 edges, 6 curated) | ✅ |
| E0 | Speech-model representation harness on GB10 GPU (WavLM, ~50 Hz) | ✅ |
| E1 | Controlled stimulus set (288 items, balanced, synthesized) | ✅ |
| E2 | H-HOLISM pre-registered experiment (initial signal) | ✅ |
| E5 | Prosody acoustic-blindness gap (speech recovers what text can't); localization null | ✅ |
| E6 | Scaling replication — CORRECTS E2 (effect does not survive) | ✅ |
| E7 | **Definitive matched-vocab re-run — settles it: NO speech-vs-text holism difference (clean null)** | ✅ |
| X0 | Autoresearch loop (reuses prabodha EFE) — full cycle; finding replicated across 4 model pairings | ✅ |

8 milestones. Run the whole ledger: `python gates/check.py`. Milestones + gates: [`specs/MILESTONES.md`](specs/MILESTONES.md).

## Discipline
Spec/PRD-driven · TDD (69 tests) · **dual-verdict gates** (code + domain, from prabodha) ·
pre-registration before results · honest negatives shipped · GPU (GB10). No fabrication, no stubs,
no bypassed gates — the Sākṣī invariant.

## Layout
```
src/pranava/         corpus/ (edition, translit, morph) · kg/ · speech/ · experiments/
data/vakyapadiya/    edition.jsonl · concept_graph.json · morph_kanda1.jsonl · coverage.json
data/stimuli/        manifest.jsonl · datasheet.json · wav/ (local)
data/experiments/    e2_results.json · e2_trajectories.png
research/            landscape scan · integration notes · prereg/ · E2-report
gates/               check.py (dual-verdict) · gate_*.json
```

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"   # + torch(cu130), transformers, piper-tts, ...
.venv/bin/python -m pytest            # 60 tests
.venv/bin/python gates/check.py       # verify all milestones
```

Built on the operator's prior GB10 work (pramana, prabodha, PWM, jSpace, Saṃsādhanī).
Lead: Sharath Sathish (arXiv 2604.04937, *Pramana*).
