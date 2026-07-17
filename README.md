# PRANAVA · Śabda-ALM

**A speech-centred Audio Language Model on a Sanskrit byte-core — and the Sphoṭa-Lens for
locating where sound becomes meaning.**

> प्रणव · *praṇava* — the primordial sound from which articulated speech and meaning unfold.

🌐 **[Project site](https://sharathsphd.github.io/pranava/)** · 🎙️ **[Try it live](https://sabda-alm.vercel.app)** · 🏛️ **[Platform architecture (NSM)](NSM.md)** · 📄 **[Paper](paper/paper.tex)** · reproduce everything with `python gates/check.py`

**Headline results** (all dual-gated):
- **Multilingual on one Sanskrit byte-core**: fed real English (LibriSpeech) *and* native Sanskrit, one 200M model transcribes both — Sanskrit CER **0.706** (beats Whisper-base 0.940), English 0.785. Capability, not English SOTA: [`research/multilingual.md`](research/multilingual.md)
- **Competitive with SOTA on authentic Sanskrit**: on native-Sanskrit speech the 200M Śabda-ALM reaches CER **0.565** — a close 2nd to NVIDIA's 0.6B Parakeet-TDT (**0.522**), well ahead of Whisper (0.754); on a controlled TTS distribution it ranks **1st** (0.546 vs 0.605). The honest two-regime read: [`research/sota-leaderboard.md`](research/sota-leaderboard.md)
- **Instruction-following from speech**: one audio clip, six tasks (transcribe / language / kartā / karaṇa / karma / kriyā) selected by an English instruction — a clean causal test (0.281 accuracy with the right instruction vs **0.000** with a shuffled one). RLAIF/DPO sharpens it (karma 0.87→**0.95**). [`research/instruction-tuning.md`](research/instruction-tuning.md) · [`research/rlaif.md`](research/rlaif.md)
- **The Sphoṭa-Lens** localises meaning-emergence-from-sound to a **validated layer 13** (correlational + causal peaks agree; meaning decodable from audio positions at 11× chance). [`research/sphota-lens.md`](research/sphota-lens.md)
- **Scales to 1.13B**: the Sphoṭa Projector trained against the 1.13B Megatron Nemotron-H core (RTX 5090) reaches multilingual CER **0.682** vs the 200M+LoRA's 0.742 — scale helps. [`research/scale-1b.md`](research/scale-1b.md)
- **Adaptation beats scaling** (within a fixed adaptation): a 200M core + 1.5M LoRA weights (CER 0.548) beats a 1.13B projector-only core (0.571) on the controlled corpus.
- Full **speech-to-speech** loop: audio → Parakeet → Sphoṭa Projector → Sanskrit core → text → TTS → audio.

---


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
| M2 | Morphological analysis via Saṃsādhanī (0.455 whitespace coverage) | ✅ |
| M2b | Sandhi segmentation (vidyut-cheda) + validation — coverage 0.455→0.533 (honest; 0.85 not reached) | ✅ |
| M3 | Verse-anchored concept knowledge graph (20 concepts, 125 edges, 6 curated) | ✅ |
| E0 | Speech-model representation harness on GB10 GPU (WavLM, ~50 Hz) | ✅ |
| E1 | Controlled stimulus set (288 items, balanced, synthesized) | ✅ |
| E2 | H-HOLISM pre-registered experiment (initial signal) | ✅ |
| E5 | Prosody acoustic-blindness gap (speech recovers what text can't); localization null | ✅ |
| E6 | Scaling replication — CORRECTS E2 (effect does not survive) | ✅ |
| E7 | **Definitive matched-vocab re-run — settles it: NO speech-vs-text holism difference (clean null)** | ✅ |
| X0 | Autoresearch loop (reuses prabodha EFE) — drove E3/E5/E6 cycles | ✅ |
| X1 | Pramāṇa epistemic layer (reuses pramana) — audits own claims; refuses the retracted holism claim as *hetvābhāsa* | ✅ |

14 milestones. See **[PAPER.md](PAPER.md)** for the consolidated write-up. Run the whole ledger: `python gates/check.py`. Milestones + gates: [`specs/MILESTONES.md`](specs/MILESTONES.md).

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
