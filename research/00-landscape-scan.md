# Landscape Scan — Session 1 (2026-07-16)

Independent research preceding vision synthesis. All claims sourced; nothing fabricated.

## Local assets confirmed first-hand

- **Vākyapadīya mūla corpus**: `docs/vakyapadiyam-various-sources/vakyapadiya_mula_deva.json` —
  1,814 entries (complete mūla, Devanagari, all 3 kāṇḍas, scraped from wisdomlib 2026-07-15), with
  per-kāṇḍa commentary JSONs in `build/commentary/kanda{1,2,3}.json` and fetch/build scripts.
  Scholarship notes "persistent gaps in comprehensive digital editions of the Vākyapadīya" — this
  corpus is therefore a genuinely scarce asset.
- **Saṃsādhanī**: Docker image `slm1/samsaadhanii:patched` (Amba Kulkarni's Sanskrit
  parsing/śābdabodha toolkit) already on this machine.
- **GPU**: GB10, native CUDA torch 2.13.0+cu130 in `pranava/.venv`; NeMo image
  `prabhasa/nemo-gb10:26.02` (46GB) for speech; `prabodha/gb10:0.1` image exists but its
  entrypoint python failed a direct exec test (arch mismatch suspected — verify before relying on it).

## External frontier (searched 2026-07-16)

### Speech/audio LMs & tokenizers
- Mimi codec (kyutai): 12.5Hz, 1.1kbps, semantic+acoustic in one tokenizer via WavLM distillation;
  Moshi full-duplex dialogue on top. [arxiv 2410.00037, hf kyutai/mimi]
- Llama-Mimi (2509.14882): interleaved semantic/acoustic tokens, single decoder, SOTA generation.
- PAST (2505.14470): phonetic-acoustic tokenizer beating SOTA on phonetic representation.
- Dynamic character-aligned speech tokenization (2601.23174): beyond fixed frames — directly
  relevant to the sphoṭa "continuum → meaning-driven segmentation" thesis.
- Benchmarks exposing the speech-vs-text semantic gap: MMSU (2506.04779, 5k QA / 47 tasks incl.
  prosody & paralinguistics), WavBench (2602.12135), ProsAudit, EmphAssess, AQUA-Bench,
  Step-Audio-R1 (2511.15848), LALM survey (2605.20266).

### Sanskrit epistemology × AI
- **Pramana paper** (arxiv 2604.04937): "Fine-Tuning LLMs for Epistemic Reasoning through
  Navya-Nyāya" — appears 3× in operator's literature_urls.txt; local project
  ~/projects/pramana likely related. MUST read + build on.
- **SHABDABODHA-LLM** (ResearchGate 407141993, very recent): automatic sentence-level śābdabodha
  generation using prakāratā–viśeṣyatā relations with LLMs. Direct competitor/collaborator work.
- Amba Kulkarni: "Sanskrit Parsing based on the theories of Śābdabodha" (2019); "Later Nyāya
  Logic: Computational Aspects" (Springer). Saṃsādhanī is her group's toolkit.
- Cognitive NLP work on Bhartṛhari (arxiv 1810.04440).

## Preliminary thesis space (to be brainstormed after digests land)

The docs' core claim: sound and awareness are directly coupled (śabdādvaita); text LLMs operate on
pre-chunked symbols and miss (a) the pre-segmental continuum, (b) verb-first/event-first ontology,
(c) a normative pramāṇa layer for testimony. Candidate build directions (NOT yet decided):

1. **Sphoṭa-bench / Śabda-bench**: a rigorous benchmark + probing suite testing whether speech LMs
   segment meaning from continuous audio the way sphoṭa theory predicts (holistic flash vs
   incremental composition) — psycholinguistics-style experiments on open speech models (Mimi/Moshi,
   SpiritLM-class), GPU-run, with falsifiable hypotheses.
2. **Vākyapadīya digital edition + knowledge graph**: the scarce-asset play — canonical, cited,
   machine-readable VP with commentary alignment, śābdabodha parses (via Saṃsādhanī), verse-level
   cross-references to modern cognitive science claims; publishable dataset.
3. **Pramāṇa-aware semantic layer**: prakāra–saṃsarga–viśeṣya structured KR extracted from
   speech/text, building on local pramana project + Pramana paper.
4. **Sphoṭa-inspired tokenizer experiment**: meaning-driven dynamic segmentation of speech
   (cf. 2601.23174) tested against fixed-frame tokenizers on GB10.

Decision criteria: verifiable milestones, GPU leverage, builds on local incomplete projects,
scholarly integrity, achievable without fabrication.
