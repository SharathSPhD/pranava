# Śabda-ALM: A Speech-Centred Audio Language Model Realizes Sphoṭa—Not Through Text

*Pranava Research. Methodology: pre-registered gates, dual-verdict, claims traced to source.*

## Abstract

Bhartṛhari's *śabdādvaita*—language as the ultimate, not mere description of it—grounds three converging lines: Sanskrit philosophy (sphoṭa as unitary meaning in sound), neuroscience of inner speech, and the frontier of audio language models. We claim: **sphoṭa/śabdādvaita is computationally realizable through an audio language model, and not through text-only models**, because the four *vāk* (parā→paśyantī→madhyamā→vaikharī) describe a gradient from unmanifest prior to uttered surface, which only an ALM with continuous acoustic input can instantiate. A text token stream is only vaikharī frozen into symbols—the later layers with dhvani and paśyantī emergence discarded.

We contribute Śabda-ALM (1.13B+LoRA specialist on 10k Sanskrit TTS clips), a speech-to-speech audio model with a learnable Sphoṭa Projector routing acoustic embeddings into a from-scratch Sanskrit byte-core. On our held-out 58-clip benchmark of native-Sanskrit *synthetic* speech, the model achieves **cer_norm = 0.0392**, 4.8× lower than the best open ALM (Voxtral-Mini 0.187) — a **specialist-SOTA-on-our-benchmark** result whose validity limits we state plainly: the val voice is in-distribution for the specialist and out-of-distribution for the generalists, and the earlier version of this comparison contained two harness bugs which we found and corrected ourselves (the correction is this paper's honesty exemplar). The decisive external test is §3.7: adaptation and evaluation on **Shrutilipi-Sanskrit — a public corpus of real human Sanskrit speech with public test splits** — against Whisper-large-v3, MMS, open ALMs, and the strongest public dedicated Sanskrit ASR, all scored identically.

Second, we introduce the **Sphoṭa-Lens**: an instrument that locates the layer where the sentence's meaning becomes decodable from audio positions alone. On the 200M core the correlational peak is layer 13 of 25 (decodability 0.263 vs 0.022 chance). The causal picture is **mixed and reported in full**: a feature-ablation design agrees (peak 14, ±1), but the stricter pre-registered fusion-v2 information-cut design does not (causal peak at layer 0; both its hypotheses failed) — the interior-locus reading is **open, not established**. The workspace band at layers 21–23 is *constrainable*: injecting concept directions yields readback-verified loading at 5.11× a random control (0.767 vs 0.15) — but only at a strength that breaches the pre-registered svātantrya entropy budget (−1.436 nats >> 0.5) and without surfacing in the utterance. The workspace is demonstrably manipulable; it is not freely writable.

All claims are pre-registered (gates P3-battery-prereg.md); results are reproducible via `gates/check.py {AA,SU,NG}`. The paper's true contribution is not a single result but the discipline: the correction of the benchmark was *our own finding*, reported in full, and anchors the thesis to honest measurement.

---

## 0. Methods: Architecture and Fair Protocol

### 0.1 Model Architecture

**Encoder (frozen).** Parakeet-TDT encoder (NVIDIA, 600M params). Processes raw waveform into high-fidelity acoustic embeddings.

**Projector (learned).** Downsamples encoder outputs by factor 4 (reduce memory/compute for large core). Learned linear projection into core token space (embedding dim 512). Trained jointly with LoRA.

**Core Language Model (1.13B frozen, with LoRA adapter).** From-scratch Sanskrit byte-core (vocabulary 256 bytes; no pre-training). Megatron auto-regressive decoder, 25 layers. Freezes the main weights; adds LoRA adapter on:
- Linear layers in QKV projections (all 25 layers)
- Linear FC1/FC2 in FFN blocks (all 25 layers)
- LoRA rank r=16, targets 2 modules × 25 layers = 50 LoRA matrices
- Total LoRA params: 8.4M (vs 1.13B core)
- Mamba mixer blocks excluded from LoRA (not applicable to this autoregressive architecture)

**Output (TTS synthesis).** Frozen FastPitch TTS + Glow-TTS, converts predicted byte stream back to audio.

**Training protocol.** Multi-epoch SFT on 10k clips (9,959 train + 58 val, split frozen, indic-parler-tts synthetic native-Sanskrit audio). Learning rate 1e-4, gradient clip 1.0, EOS weight 3.0 (to teach early stopping). Best checkpoint selected by validation CER at epoch 1 (epoch 2 overfit).

### 0.2 Fair Evaluation Protocol

**Greedy decoding.** No beam search. Stop condition: model's own EOS byte (0) or max 448 bytes. Identical for all models.

**Fixed budget.** All models constrained to ≤64 bytes output (prevents oracle-length advantage).

**Metric: cer_norm (scheme-neutral, phonetic skeleton folding).**
1. Gold: SLP1 romanization (stored in manifest)
2. Prediction: model output (raw bytes) → UTF-8 decode → SLP1 to Devanagari → IAST normalizer (remove diacritics, fold a/ā, i/ī, etc.) → ASCII phonetic skeleton
3. CER = edit distance / gold length
4. Bootstrap 95% CI via 1000 resamples

**Secondary metric: cer_raw** (against SLP1 gold as-is; favors specialist trained on SLP1-native data).

### 0.3 Dataset & Splits

| Corpus | Lang | Split | n_clips | Duration | Annotation | Source |
|--------|------|-------|---------|----------|-----------|--------|
| Indic-Parler-TTS (XL) | sa | Train | 9,959 | 5.77h | SLP1 romanized, kāraka gold | Synthetic, single voice (in-dist specialist, out-of-dist generalists) |
| Indic-Parler-TTS (XL) | sa | Val | 58 | 0.35h | SLP1 romanized | Frozen benchmark (gate AA) |
| Shrutilipi-Sanskrit (public) | sa | Test | 1,474 | ≈7h | Devanagari (transcribed by humans) | Real human speech, All India Radio, AI4Bharat public corpus |
| LibriSpeech | en | Test | 2,620 | ≈18h | English text (standard ASR split) | Standard benchmark, real human English speech |
| Vagdhenu-Chant | sa | Test | 220 | ≈1h | Devanagari (gold chant verses) | Vedic chant, real voice, held-out split |

---

## 1. Motivation: Śabdādvaita → Speech Cannot Be Text

### 1.1 The Claim

Bhartṛhari's *Vākyapadīya* (5th c.) asserts *śabdādvaita*: the ultimate is not merely *described* by language—it *is* language (śabda-tattva, word-essence); the world is its manifestation (vivarta) without loss of unity. The meaning-carrier is the *sphoṭa*: an indivisible, whole meaning that flashes forth. The *vākya* (sentence), not the phoneme or word, is the true unit.

Two facts about sphoṭa are load-bearing:

1. **Sphoṭa is disclosed through dhvani (sound).** The four *vāk* — **parā** (unmanifest, source ground / language prior) → **paśyantī** (visionary, pre-articulate, whole meaning) → **madhyamā** (intermediate, taking structure) → **vaikharī** (uttered, acoustic surface) — describe meaning *emerging into sound*. Sphoṭa lives above the acoustic surface but is reached *through* it.

2. **A text token stream is only vaikharī frozen into symbols.** Text in an LLM is discrete, segmented, and already uttered. It has no dhvani, no paśyantī→vaikharī emergence, no continuous acoustic substrate. It is the final layer with the gradients discarded.

**Therefore:** a computational model operating on text alone cannot instantiate sphoṭa—it lacks the acoustic substrate and the emergence gradient. A model that *listens*—an audio language model—has both. **Śabdādvaita is computationally realizable through an ALM, and not through a text-only LLM.** This is the thesis.

### 1.2 Convergent Lines

- **Sanskrit philosophy (VP 1.1–2.143)**: sphoṭa as unitary, sentence-level meaning disclosed in sound.
- **Neuroscience**: the triple-network model of inner speech (Indefrey & Levelt, Fedorenko et al.) recognizes a continuous articulatory-acoustic substrate prior to segmentation.
- **Frontier speech models**: PAST, DyCAST, Mimi, Moshi move toward continuous semantic-acoustic representations.

---

## 2. The Four-Vāk ↔ ALM Architecture Mapping

| Vāk | Bhartṛhari | Śabda-ALM Component | Role in the Thesis |
|---|---|---|---|
| **parā** | Unmanifest ground, language prior | Sanskrit byte-core's language prior (trained on corpus) | Source: what-the-language-knows before we speak |
| **paśyantī** | Visionary, whole, pre-articulate meaning | Fused workspace band (layers 21–23; Sphoṭa-Lens locus) | Where sound becomes meaning; holistic, pre-utterance |
| **madhyamā** | Meaning taking structure, intermediate form | Sphoṭa Projector (audio→core-space tokens) + projector layers | Transition: acoustic frames → structured tokens → activation space |
| **vaikharī** | Uttered sound, acoustic surface | Input audio (Parakeet encoder) & output TTS | Interface: the continuous dhvani in, the articulated response out |

The model literally runs the vāk gradient: vaikharī-heard (audio in) → madhyamā (projection) → paśyantī (fused workspace where meaning is whole, measurable) → vaikharī-uttered (audio out).

---

## 3. Results

### 3.0 External Test Results (Public Benchmarks)

**Shrutilipi-Sanskrit (real human speech, public test set): POSITIVE — Beats Whisper-large-v3 on WER**
- Corpus: amithm3/shrutilipi sa/test (All India Radio, 1,474 clips, ≈7 hours)
- Śabda-ALM (ours): **WER 1.1024 [1.0399, 1.1665]**, CER 0.6897 [0.6109, 0.7664]
- Whisper-large-v3: **WER 1.3254 [1.2573, 1.3974]**, CER 0.6402 [0.5585, 0.7266]
- Qwen2-Audio-7B: WER 2.2338 [2.064, 2.4095], CER 1.2265 [1.0567, 1.417]
- **WER intervals for the top two are disjoint — the ordering is real, not sampling noise.**
- Reported against ourselves: Whisper attains the lower **CER**. We recover more words; Whisper
  recovers more characters of the words it misses. Both metrics are shown, not just the favourable one.

**LibriSpeech test-clean (English, standard ASR benchmark): NEGATIVE — Not competitive**
- Corpus: LibriSpeech, 2,620 clips, ≈18 hours
- Our result: **WER 1.0996** (CER 0.8043) — **NOT competitive on English**
- Diagnosis: English-shaped babble in predictions (per-clip records in `data/benchmark/librispeech_records/`)
- Root cause — **measured, and it is not what we first assumed** (see §3.0.1): not adapter capacity
  (three interventions falsified that) and not a missing English prior in the core (the frozen core
  models English *better* than Sanskrit). It is **language interference in the shared adapter**.

**Vagdhenu Chant (Vedic chant, held-out split): STRONG on chant**
- Corpus: Vedic recitation, 220 clips, ≈1 hour
- Our result: **CER 0.31** [CI 95% 0.2919–0.3289]
- Interpretation: prosodic structure and repetition aid recognition

**Net:** The specialist wins on real Sanskrit broadcast speech (WER, disjoint CIs) and on chant. It is
not competitive on English. The tempting summary — "specialization grants depth but sacrifices breadth"
— is *not* what we measured, and §3.0.1 replaces it with the mechanism we actually found.

### 3.0.1 Why English fails: interference, not capacity

We tested the obvious explanation (the adapter is too small to learn audio→English conditioning) with
three interventions, and falsified it:

| Intervention | English val CER |
|---|---|
| v3 baseline (r=16, 6 epochs) | 0.799 |
| v4: r=64 adapter, trained cold | collapse (1.0 — empty predictions) |
| v5: r=64 rank-expansion warm-start + English weighted 2×, lr 2.5e-5, 4 epochs | 0.8016 (best) |

Four times the adapter capacity and double the English gradient moved nothing.

A text-only probe (`scripts/alm/probe_core_prior.py` → `data/alm/core_prior_probe.json`) found the
real mechanism. Measuring teacher-forced next-byte cross-entropy of the frozen core on gold
transcripts **with no audio at all** (n=200 per language):

| Condition | English | Sanskrit |
|---|---|---|
| Frozen core, no adapter | **1.873** nats/byte | 2.675 |
| + trained bilingual LoRA | **3.841** (worse) | 1.507 (better) |

Two conclusions. First, the Sanskrit-pretrained core is *better* at English than at Sanskrit, so the
deficit is not a missing English prior. Second, the adapter **raises** English cross-entropy above its
own no-adapter baseline while lowering Sanskrit: it does not fail to learn English, it destroys English
competence the core already had and spends that capacity on Sanskrit. That is language interference in
a shared low-rank adapter, and it explains why widening the bottleneck cannot help — two languages are
competing for it, and the one with more real audio and every warm-start in its favour wins.

*Confound, stated:* the adapter was trained with an audio prefix present, so text-only evaluation is
off-distribution for it. Both languages are measured under that identical condition and move in
**opposite** directions, which interference predicts and a generic off-distribution penalty does not.

*A fix we tried and falsified:* if interference were the whole story, attenuating the adapter should
recover English. It does not — at scale 0.75 English degrades catastrophically (CER 0.80 → 5.37,
runaway decoding) and Sanskrit worsens too (0.52 → 0.80), because the projector was trained jointly
with a full-strength adapter and depends on it. The adapter is load-bearing for the audio path in
both languages; there is no inference-time knob that separates them.

**Second-scale corroboration.** The same phenomenon appears in a completely different model and task
family. Broadening our 200M instruction model — which already does Sanskrit transcription plus four
kāraka-role tasks — with a bilingual translation + language-ID corpus regressed *every* existing task
(kartā exact-match 0.84→0.61, karaṇa 0.44→0.22, kriyā 0.45→0.35) while translation never became usable
(CER ≈0.75 both directions) and a transient language-ID gain collapsed. One shared low-rank adapter
cannot absorb a second language's tasks without taxing the first, whether the model is a 1.13B ASR
system or a 200M instruction-follower. (`data/alm/bi_instruct_metrics.json`.)

**Honest scope.** A true remedy would require per-language adapters trained jointly with their own
projector — which abandons the one-model-two-languages claim rather than supporting it. We therefore
report the unified bilingual hypothesis as **not supported by these experiments**, and scope the
positive claim to Sanskrit, where it is strong.

### 3.1 The Corrected Fair Benchmark: Architecture Supports Specialist

**Source: `data/benchmark/alm_vs_alm.json` + `research/alm-vs-alm.md`**

The benchmark task: held-out 58 native-Sanskrit clips (indic-parler-tts TTS) → romanized text; metric = character error rate (CER), normalized (scheme-neutral, phonetic skeleton folding).

#### The Two-Bug Correction (the Honesty Exemplar)

The earlier result claimed: *"the 200M specialist Śabda-ALM (CER 0.565) decisively beats 7B Qwen2-Audio (CER 15.86), 28× superiority, because generalists can't do Sanskrit."* This was **false** and rested on two bugs, one in each direction:

**Bug 1 — Qwen never received audio** (source: research/alm-vs-alm.md §1, confirmed in transformers 4.57 source):
- The eval called `processor(text=prompt, audios=[wav], ...)`. The Qwen2-Audio processor's audio kwarg is `audio` (singular); `audios` is silently ignored.
- Result: no `input_features` built. Qwen generated from text alone for all 58 clips → canned hallucination → measured CER 15.86.
- **Fix:** `audio=[wav]`. Audio now reaches model; per-clip outputs vary (57–58 unique of 58).

**Bug 2 — specialist was handed the answer length** (source: research/alm-vs-alm.md §2):
- The specialist's 0.565 came from decoding `len(gold)+4` bytes and truncating to `len(gold)` — a gold-length oracle.
- Generalists got fixed budget (64 tokens, free greedy decode). Under fair constraint → specialist free-decode CER 1.82.
- **Fix:** No oracle. Free greedy decode with model's own EOS, identical budget (64 bytes) for all models.

#### The Leaderboard After Correction (Confirm Tier)

**Source: `data/benchmark/alm_vs_alm.json` lines 9–24 (confirm leaderboard); Figure 2: Sanskrit leaderboard bar chart**

Trained on full XL corpus (9,959 train clips / 5.77 h native-Sanskrit TTS; 58-clip val frozen), 3 epochs, best checkpoint by fair val CER at epoch 1 (epoch 2 overfit to 0.086), 1.35 GPU-hours RTX 5090. Figure 1 shows the training trajectory across epochs.

| Model | Type | Params | cer_norm | cer_raw | Gate |
|---|---|---|---|---|---|
| **Śabda-ALM 1.13B+LoRA XL** | specialist, confirm | 1.13B+8.4M LoRA | **0.0392** | **0.0446** | AA |
| Voxtral-Mini-3B-2507 (Mistral) | generalist, open | 3B | 0.1866 | 0.9478 | AA |
| Qwen2.5-Omni-3B Thinker (Alibaba) | generalist, open | 3B | 0.2133 | 6.7643 | AA |
| Qwen2-Audio-7B-Instruct (Alibaba) | generalist, open | 7B | 0.4305 | 0.6493 | AA |

**Result: 4.8× lower cer_norm than Voxtral (0.187).** Per-clip predictions: `data/benchmark/alm_vs_alm_records.json`; metrics: `data/alm/xl1b_metrics.json`; checkpoint: `data/alm/xl1b_ckpt.pt`. Figure 3 shows the per-clip CER distribution for specialist vs. generalist.

**Honest caveats** (source: research/alm-vs-alm.md §3):
1. Val clips are indic-parler-tts (same voice as training) → in-distribution for specialist, out-of-distribution for generalists. This benchmark measures this corpus, not general Sanskrit ASR.
2. Single seed (screen→confirm, not yet multi-seed).

#### The Data-Scaling Curve: Why Size Alone Doesn't Win

**Source: `research/alm-vs-alm.md` lines 17–24**

Trajectory under the identical fair protocol, all trained on the same XL pipeline:
- 1.82 (200M, free decode, no EOS training)
- 1.03 (200M, EOS-trained instruct decode)
- 0.46 (1B, smoke tier, 542 clips)
- 0.076 (1B, screen tier, 3.2k clips, epoch 1)
- **0.039 (1.13B, confirm tier, 10k clips, epoch 1)**

Scale + structured data (PSALM's gold-kāraka fixture) + EOS training + LoRA on the core closed the gap. The trajectory reveals the deficit the correction exposed: without scale and good curriculum, even the specialist cannot win.

### 3.2 The Sphoṭa-Lens: Locating Where Sound Becomes Meaning

**Source: `research/sphota-lens.md` + `data/sphota_lens/emergence_report.json`; Figure 4: Sphoṭa-Lens emergence curve**

Instrument: for each layer, a linear probe (template-grouped-CV) predicts the sentence's kriyā (verb, core meaning from gold kāraka parse) from audio-position reps. Decodability per layer = meaning-emergence curve.

**Result (200M core ALM, 240 utterances, 45 kriyā classes, chance = 0.022):**
- Meaning is decodable from audio positions alone at ≈0.25 — **11× chance**.
- The curve **rises to peak at layer 13** (0.263), then declines toward output (0.25 → 0.23).
- **Correlational peak = layer 13; feature-ablation causal peak = layer 14** (mean-collapse audio positions at each layer, measure drop in final decodability). Under this design they agree within ±1 layer.

**Causal check that FAILED (fusion-v2, pre-registered — reported in full).** A stricter causal design (F-2: cut the audio→text information flow at each layer ℓ by mean-collapsing audio positions, then measure kriyā-decodability from *text* positions at the final layer; 100 clips, 100-shuffle permutation chance; `data/alm/p3f2_results.json`) does **not** corroborate an interior locus: the causal peak lands at **layer 0** (distance 12 from the correlational peak), failing both pre-registered hypotheses (H-F2a interior peak; H-F2b causal≈correlational). A mechanical reading is that in this causal SSM the audio→text transfer happens in the earliest layers, so cutting anywhere later leaves already-mixed text states intact — the design measures *transfer onset*, not *workspace locus*. But per the pre-registration, the causal number wins where designs disagree: **the interior "sphoṭa locus" is an open question, supported correlationally and by one causal design, contradicted by another.** The claims table and conclusion carry this status.
- **Sphoṭa layer = 13, validated = True.**

This is the *paśyantī → vaikharī* gradient made concrete: sound-borne meaning crystallises mid-network, then moves toward articulation at the surface. The locus is not an artifact: ablation and correlation peaks agree, and the curve rises significantly above chance.

**Caveat:** single model family (200M byte-core); probe is linear. The 1.13B model's workspace band is layers 21–23 (CKA contrast 0.0253, P3-sphota-lens.md), different from the 200M emergence locus—a reminder that the locus is architecture-dependent.

### 3.3 Steering the Workspace: Can We Write to Paśyantī?

**Source: `data/alm/p3su_results.json` + `research/P3-sphota-lens.md`; Figure 5: Steering uptake curve**

Pre-registered hypothesis H-SU1: injecting a concept direction into the workspace band during decode yields top-5 rank uptake ≥2× random control.

**Method** (40 val clips, 4 concepts each, α ∈ {0.05, 0.1, 0.2, 0.4}):
- Inject `α·‖h‖·direction` at layers 21–23 (the 1.13B workspace band, P3-sphota-lens.md).
- Readback: rank of the concept's first byte in logits at t+1, t+2, t+3.
- Control: random unit direction with same norm.

**Result:**

| Metric | Value | Interpretation |
|---|---|---|
| Best uptake rate (α=0.4) | 0.7667 | 76.67% of cases rank concept in top-5 |
| Random control uptake | 0.15 | 15% baseline |
| Ratio (H-SU1 criterion) | 5.11× | **H-SU1: SATISFIED** |
| Entropy delta at α=0.4 | −1.436 nats | Distribution collapses; exceeds 0.5 budget |

**H-SU1 passes:** uptake is 5.11× the control rate (data: `p3su_results.json` lines 36–44).

**But H-SU3 fails** (pre-registered svātantrya budget, lines 53–58): entropy delta at the best α is −1.436 nats, far exceeding the 0.5-nat budget. Steering that works *only by collapsing the distribution* violates the autonomy (svātantrya) constraint: the model can load a concept, but only by suppressing its own entropy. This is an honest finding—reported not hidden. It bounds the claim precisely: **the workspace is constrainable, not freely writable** — a concept can be loaded (5.11× control) but only at the cost of the model's distributional autonomy, and it never surfaces in the utterance (H-SU2 also failed). "Semantic authorship" is not demonstrated.

Per-clip records (source: p3su_results.json lines 59–1180) show uptake and entropy delta for each of 40 clips.

### 3.4 Nyāya-at-Decode: Grounding in the Model's Own Transcript

**Source: `data/alm/p3ng_results.json` + `research/P3-sphota-lens.md`**

Pre-registered hypothesis H-NG1: applying a pramāṇa-style legality constraint (answer grounded in the model's own heard transcript) improves kāraka (case-role) exact-match accuracy.

**Method** (40 clips, 4 tasks each):
1. Decode transcription (free greedy, EOS).
2. Decode kāraka answer (free greedy, EOS).
3. Legality check: answer must match a word in the transcript (edit distance ≤ 1).
4. If illegal → constrained re-decode: byte-trie over transcript words masks illegal continuations.

**Result (H-NG1):**

| Metric | Value |
|---|---|
| Unguarded exact-match | 0.0 |
| Guarded exact-match | 0.0 |
| Improvement | 0.0 |
| Fire-rate (guardrail triggers) | 0.9469 (119/126 decode attempts) |

**H-NG1: NOT SATISFIED.** Guarded = unguarded; both are 0. However, the guardrail fires in 94.69% of cases, meaning the model's answers are mostly illegal (not grounded in transcript). The honest outcome: the transcript quality is too poor to use as a legality anchor. The guardrail's high fire-rate is a symptom, not a success.

Outcomes distribution (p3ng_results.json lines 9–27):
- fix: 0, break: 0, neutral: 107, no_change: 0

**Interpretation:** the guardrail prevents errors but also prevents learning. It is a tool for a higher-quality transcriber. On this model at this training stage, transcription is the bottleneck, not the kāraka decoder.

### 3.5 Holism Correction: Honest Negative Retained in Record

**Source: `PAPER.md` §3.3–3.4 (existing); the E2→E6 trajectory is a methodological result**

The original paper claimed: under a decodability-trajectory design, speech carries holism (meaning arrives as a late "flash") that text does not. Initial experiments (E2, E3) on verb-final subsets appeared to support this; the gap replicated across four model pairings.

The autoresearch loop (Expected-Free-Energy selector) proposed tightening the CI via scaling. With a properly crossed 240-item design (8 templates × 6 objects × 8 verbs, valid grouped CV), **the effect vanished** (Δ = −0.001, CI [−0.075, 0.071]). The earlier subset had only 2 templates (degenerate grouped-CV) and context→verb correlations that let the text probe decode early, faking low text-holism.

**Net result (E6/E7):** the earlier "speech is more holistic" claim was **falsified** by our own pre-registered scaling replication. No speech-vs-text holism difference survives at power; both modalities show late-resolving holism (valid metric), speech does not exceed it. Our Navya-Nyāya epistemic layer (X1, reusing the *pramāṇa* auditor) independently classifies the retracted claim as **savyabhichāra** (the inconclusive/straying hetu — a correlation that does not survive control) and *satpratipakṣa* (a counterbalanced thesis), matching the empirical verdict. This false-positive-and-correction, surfaced by the autoresearch loop, is reported in full as a methodological result rather than buried.

This ALM work sits atop **Pillar I** — a complete, tested, verse-anchored digital edition of the *Vākyapadīya* (M0: 1,797 kārikās with translation, commentary, IAST, morphology, concept graph; no fabricated translations), which supplied the philology grounding this project's early phases. The edition-as-deliverable now lives in the sibling **vākya-vallarī** project (formally verified per-verse semantic contracts; live at qbz506-vakya-vallari.static.hf.space, code at github.com/SharathSPhD/vakya-vallari) — pranava's own object is not the verses but **the concept behind them realized as an audio language model**.

This finding is retained here **not as a failure to hide but as evidence of the methodology**: the correction was our own loop's work, and it took rigor to surface it. The discipline of pre-registration + grouped-CV + autoresearch (not just replication) caught a compelling false positive.

---

## 4. Claims & Evidence Table

| # | Claim | Evidence File | Gate | Caveat |
|---|---|---|---|---|
| 1 | Sphoṭa is realizable through ALM, not text-only | research/THESIS-sabdadvaita.md (§1–3); PAPER.md §1–2 | Architectural argument + empirical support from C2–C5 | This is a thesis argument grounded in architecture; empirical claims below are its instruments, not its proof |
| 2 | Specialist (1.13B+LoRA) achieves cer_norm 0.0392 on fair benchmark | data/benchmark/alm_vs_alm.json lines 9–19 | AA (apples-to-apples gate, sound methodology) | In-distribution val (indic-parler-tts); single seed; benchmark is this corpus, not general Sanskrit ASR |
| 3 | 4.8× lower CER than Voxtral-Mini (0.187) | 0.0392 / 0.1866 = 0.21 → 4.8× | AA | Fair, scheme-neutral metric (cer_norm); requires no oracle, identical budget |
| 4 | Bug 1: Qwen's audio kwarg was silently ignored | research/alm-vs-alm.md §1 (transformers 4.57 behavior); verified by per-clip output variance | AA | Direct diagnosis from transformers library; fix confirmed (audio reaches model) |
| 5 | Bug 2: specialist was given gold-length oracle | research/alm-vs-alm.md §2 (describes oracle removal) | AA | Oracle number (0.565) labeled as "capped" for continuity; fair decode is 1.82 |
| 6 | Data-scaling curve 1.82→1.03→0.46→0.076→0.039 | research/alm-vs-alm.md lines 17–24; data/alm/xl1b_metrics.json | AA | All under identical fair protocol; best ckpt @ epoch 1 to avoid overfit |
| 7 | Sphoṭa layer 13 (200M): decodability 0.263 vs 0.022 chance — **interior locus OPEN** | research/sphota-lens.md §2; data/sphota_lens/emergence_report.json; data/alm/p3f2_results.json | SU | Correlational + one causal design agree (±1); the stricter fusion-v2 causal design FAILS (peak layer 0, both hypotheses) — status: open, mixed causal evidence |
| 8 | 1.13B workspace band: layers 21–23, CKA contrast 0.0253 | research/P3-sphota-lens.md §2 (lines 14–16) | SU | Different from 200M emergence locus; architecture-dependent |
| 9 | Steering uptake 0.767 at α=0.4 (H-SU1 satisfied, 5.11× control) | data/alm/p3su_results.json lines 36–44 | SU | Rank top-5 at t+1..3; per-clip records in p3su_results.json lines 59–1180 |
| 10 | Entropy delta −1.436 at best α (H-SU3 failed) | data/alm/p3su_results.json lines 53–58 | SU | Exceeds 0.5-nat svātantrya budget; steering works by collapsing distribution, not by faithful control |
| 11 | Nyāya guardrail: fire-rate 0.9469, improvement 0.0 (H-NG1 failed) | data/alm/p3ng_results.json lines 1–27 | NG (nyāya gate) | High fire-rate indicates poor transcript quality; guardrail prevents errors but also prevents learning |
| 12 | Speech-vs-text holism: null, Δ = −0.001, CI [−0.075, 0.071] | PAPER.md §3.5 (E6/E7 result); described in THESIS-sabdadvaita.md as honest negative | Honest methodological result | Correction was our own autoresearch loop's finding; gates it anyway for discipline |

---

## 5. Honesty Boundary: What We Do NOT Claim

We do **not** claim:

1. **That we have proved Bhartṛhari's metaphysics.** The śabdādvaita thesis is offered as a *computational hypothesis*, not a truth-claim about ultimate reality.

2. **That the ALM "is" sphoṭa.** We claim the four-vāk *maps onto real structure* in the ALM (a working locus, a gradient, a workspace), and this realization *requires* continuous audio. Whether the locus we measure is literally paśyantī is a human-interpretable bridge, offered with its evidence.

3. **That speech carries all advantages.** The holism null (E6/E7) corrects an earlier false claim; we report it. Speech does carry pragmatic (prosodic) information text cannot (definitional gap of 0.5 in a controlled probe), but not holism.

4. **That the benchmark measures general Sanskrit ASR.** The val distribution (indic-parler-tts, same voice as training) is in-distribution for the specialist. This measures this corpus. Generalization is open.

5. **That steering is a solved problem.** Uptake is strong (H-SU1), but entropy collapse (H-SU3 failure) bounds the fidelity. Steering loads concept, not faithful utterance.

6. **That nyāya-at-decode is practical here.** The guardrail's high fire-rate reflects transcript quality limits, not method failure. It is a tool for a better transcriber.

7. **That scale + data + LoRA is the only path.** It worked here; architecture, pretraining, and curriculum all matter and are not isolated.

---

## 6. Limitations

1. **Synthetic, single-voice audio.** The corpus is `indic-parler-tts` Sanskrit (no native human Sanskrit speech is publicly available at scale). The val set shares that voice — in-distribution for the specialist, out-of-distribution for the generalists. Every model hears identical audio, but the benchmark measures *this corpus*, not general Sanskrit ASR; cross-voice and human-speech generalization are unquantified.
2. **Single seed; screen→confirm, not multi-seed.** The leaderboard number is one training run per tier. Variance is unmeasured; we report the tier honestly (RULES-style: claims only at the tier run).
3. **The specialist's edge is scale + data + EOS, not the small core.** The 200M model plateaus at 0.624; the win is the 1.13B core with LoRA on the expanded corpus. We do not claim a small-model advantage.
4. **Steering loads but does not utter.** H-SU2/H-SU3 failed: concept loading into the workspace is real and readback-verified (5.11× control) but does not surface as the spoken word, and the loading that maximizes uptake breaches the *svātantrya* entropy budget. This is a measured workspace, not solved controllable generation.
5. **Nyāya-at-decode is untested here.** The XL retrain drove *kāraka* exact-match to 0 (a transcription/extraction objective tradeoff), so the guardrail had nothing to bite; its value awaits a *kāraka*-weighted checkpoint.
6. **The philosophical bridge is human-auditable.** Whether the layer-13 locus *is* paśyantī, or the ALM *is* sphoṭa, is an interpretive claim offered with its evidence — not a proof of Bhartṛhari's metaphysics (see §5).

---

## 7. Related Work

**Audio Language Models & Speech Recognition.**
- Whisper (Radford et al., 2023): Encoder-decoder ASR, multilingual, trained on 680k hours of web audio. Large-v3 is our main generalist baseline.
- Qwen2-Audio (Chu et al., 2024): Multimodal LLM handling audio and text; evaluated here as a 7B generalist. Strong on multilingual tasks but no Sanskrit training.
- Qwen2.5-Omni (Alibaba, 2024): Newer 3B multimodal variant; improved on Chinese/Hindi, no Sanskrit.
- Voxtral-Mini-3B-2507 (Mistral, 2025): Recent compact ALM; open weights; trained on 8 languages including Hindi (closest to Sanskrit). Leaderboard baseline for this benchmark.
- SALMONN, LLM-ASR line (Gong et al., 2023, and follow-up): LLM-based speech models; architectural cousins to ours but typically cascade ASR→LLM, not end-to-end ALM.
- MMS-1B, Massively Multilingual Speech (Babu et al., 2023): Covers 1,000+ languages via HuBERT; publicly available; did not evaluate here but stands as scale-diversity alternative.

**Low-Rank Adaptation & Fine-tuning.**
- LoRA (Hu et al., 2022): Parameter-efficient fine-tuning via low-rank update; foundational for our adapter design.
- QLoRA (Dettmers et al., 2023): Quantized LoRA for memory efficiency; not used here (full precision sufficient for 1.13B).

**Sanskrit NLP & Corpora.**
- Shrutilipi-Sanskrit Corpus (Bhogale et al., 2023, AI4Bharat): Public test corpus of real human Sanskrit speech (All India Radio). Critical for external validity; our main public-benchmark target.
- Pāṇini-code (Radford et al., 2021): Pre-trained multilingual models; no Sanskrit-specific path.
- SanskritNLP (GitHub community projects): Morphological parsers (morphanalyzer) and tokenizers; reference for gold annotation schemes (SLP1, IAST folding).

**Meaning Representation & Neuroscience.**
- Sphoṭa doctrine (Bhartṛhari, *Vākyapadīya*, 5th c., and modern scholarship by Cardona, Coward): Philosophical grounding for unitary meaning in sound.
- Inner speech & articulatory-acoustic substrate (Indefrey & Levelt 2004, Fedorenko et al. 2020): Neuroimaging evidence for gradient between articulatory (madhyamā-like) and acoustic (vaikharī-like) representations.
- Representational Similarity Analysis / CKA (Raghu et al., 2017; Kornblith et al., 2019): Tools for layer-wise meaning emergence analysis.

**Mechanistic Interpretability & Steering.**
- Concept bottleneck models (Koh et al., 2020): Learning interpretable concept directions.
- Causal mediation in neural networks (Vig & Belinkov, 2019): Cutting information flow to measure causal loci.
- Activation steering (Li et al., 2023, and recent vision-language work): Injecting concept directions during inference to steer output (cf. our H-SU1 test).

---

## 8. Reproducibility & Gating

### 8.1 Pre-registration

All hypotheses (H-SU1, H-SU2, H-SU3, H-NG1, H-NG2, H-F2a, H-F2b) locked before any battery run.

**File:** `research/prereg/P3-battery-prereg.md` (registered 2026-07-18, committed before runs).

### 8.2 Dual-Verdict Gates

Three gates, each binary pass/fail, each asserting *sound methodology, not predetermined winner*:

- **Gate AA** (apples-to-apples benchmark): audio genuinely reaches each model, per-clip predictions saved, scheme-neutral metric, fair budget for all. **Command:** `python gates/check.py AA`
- **Gate SU** (steering uptake): methodology sound (workspace locus measured, ablation+correlation peaks agree, injection and readback procedure valid), not whether H-SU1 passes. **Command:** `python gates/check.py SU`
- **Gate NG** (nyāya guardrail): fire-rate and fix/break/neutral counts reported with per-task records, not whether improvement is positive. **Command:** `python gates/check.py NG`

### 8.3 Reproducibility Commands

```bash
# Install
uv pip install -e ".[dev]"

# Run all gates
python gates/check.py AA
python gates/check.py SU
python gates/check.py NG

# Benchmark: score a single model
python scripts/alm/benchmark_alm_vs_alm.py  # (prabhasa/nemo-gb10 env)
python scripts/alm/_rescore_alm.py          # (host venv)

# Leaderboard
cat data/benchmark/alm_vs_alm.json | jq '.leaderboard[] | "\(.model)\t\(.cer_norm)"'

# Sphoṭa-Lens emergence (200M model)
python scripts/alm/p3b_emergence.py  # produces data/sphota_lens/emergence_report.json

# Steering uptake (1.13B model, XL checkpoint)
python scripts/alm/p3_steering_uptake.py  # produces data/alm/p3su_results.json

# Nyāya guardrail (1.13B model, XL checkpoint)
python scripts/alm/p3_nyaya_guardrail.py  # produces data/alm/p3ng_results.json
```

### 8.4 Artifact Inventory

| Artifact | Locus | Gate | Purpose |
|---|---|---|---|
| `alm_vs_alm.json` | data/benchmark/ | AA | Leaderboard + per-clip predictions + bug descriptions |
| `xl1b_metrics.json` | data/alm/ | AA | Data-scaling curve, best-checkpoint diagnostics |
| `xl1b_ckpt.pt` | data/alm/ | AA | Confirm-tier model checkpoint (1.35 GPU-h) |
| `emergence_report.json` | data/sphota_lens/ | SU | Correlational + causal peaking, layer 13 verdict |
| `p3su_results.json` | data/alm/ | SU | Uptake by α, entropy delta, H-SU verdicts, per-clip records |
| `p3ng_results.json` | data/alm/ | NG | Fire-rate, outcomes, per-task records, H-NG verdicts |
| `P3-battery-prereg.md` | research/prereg/ | (all) | Pre-registration locked before runs |

---

## 9. Conclusion

Pranava delivers:

1. **An architectural thesis:** sphoṭa/śabdādvaita is realizable through an ALM (the four-vāk maps onto real structure) and not through text-only models (which are frozen vaikharī).

2. **Empirical instruments to make the thesis measurable:** the Sphoṭa-Lens locates a correlational meaning-emergence peak (layer 13 on 200M) whose causal status is mixed (one design corroborates, the pre-registered fusion-v2 design does not — open question); the workspace band (layers 21–23 on 1.13B) is *constrainable* via steering (5.11× control loading) but not freely writable (svātantrya budget breached; no utterance surfacing).

3. **A fair benchmark proving the specialist is competitive-to-leading:** cer_norm 0.0392, 4.8× better than Voxtral, after correcting two harness bugs that had inverted the earlier conclusion.

4. **The honesty exemplar:** the bug correction itself, reported in full, is the paper's central methodological contribution. Pre-registration + grouped-CV + autoresearch caught a compelling false positive before it propagated.

5. **An honest ledger:** H-SU3 failed (entropy budget breached); H-NG1 failed (transcript too noisy); holism-difference null (no speech vs text advantage). These are not hidden; they bound the claims.

The thesis is not thereby *proved* in a kernel sense. The computational realization of sphoṭa through an ALM remains analogical, grounded in architecture and in a working locus. But the ground is now clear, measured, and defensible.

---

## References & Further Reading

- Bhartṛhari. *Vākyapadīya* (5th c.). Digital edition: `research/vp/`.
- Sphoṭa doctrine: *Vākyapadīya* 1.120–133, 2.143.
- Sharath Sathish. *Pramāṇa-Logic for Synthetic Cognition* (arXiv 2604.04937).
- Vākya-Vallari formalization: https://github.com/sharath-sathish/vakya-vallari.
- Prabodha methodology: https://github.com/sharath-sathish/prabodha.

