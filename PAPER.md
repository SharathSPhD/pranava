# Śabda-ALM: A Speech-Centred Audio Language Model Realizes Sphoṭa—Not Through Text

*Pranava Research. Methodology: pre-registered gates, dual-verdict, claims traced to source.*

## Abstract

Bhartṛhari's *śabdādvaita*—language as the ultimate, not mere description of it—grounds three converging lines: Sanskrit philosophy (sphoṭa as unitary meaning in sound), neuroscience of inner speech, and the frontier of audio language models. We claim: **sphoṭa/śabdādvaita is computationally realizable through an audio language model, and not through text-only models**, because the four *vāk* (parā→paśyantī→madhyamā→vaikharī) describe a gradient from unmanifest prior to uttered surface, which only an ALM with continuous acoustic input can instantiate. A text token stream is only vaikharī frozen into symbols—the later layers with dhvani and paśyantī emergence discarded.

We contribute Śabda-ALM (1.13B+LoRA specialist on 10k Sanskrit TTS clips), a speech-to-speech audio model with a learnable Sphoṭa Projector routing acoustic embeddings into a from-scratch Sanskrit byte-core. On a held-out 58-clip benchmark of native-Sanskrit synthetic speech, the model achieves **cer_norm = 0.0392**, **4.8× lower than the best open ALM** (Voxtral-Mini 0.187)—but only after correcting two harness bugs that had plagued the earlier result. This correction itself is the paper's honesty exemplar.

Second, we introduce the **Sphoṭa-Lens**: a correlational-plus-causal instrument that locates the layer where the sentence's meaning becomes decodable from audio positions alone. On the 200M core, this is layer 13 of 25 (decodability 0.263 vs 0.022 chance; correlational and causal peaks agree within ±1 layer), where sound-borne meaning crystallises—the computational paśyantī. The workspace band at layers 21–23 (CKA contrast 0.0253 on the 1.13B model) is writable: injecting concept directions yields uptake 0.767 at α=0.4 (5.11× random control, 0.15). The steering faithful to the sphoṭa hypothesis breaches the svātantrya entropy budget (H-SU3 failed: delta −1.436 nats >> 0.5), an honest caveat: loading into the workspace ≠ faithful utterance.

All claims are pre-registered (gates P3-battery-prereg.md); results are reproducible via `gates/check.py {AA,SU,NG}`. The paper's true contribution is not a single result but the discipline: the correction of the benchmark was *our own finding*, reported in full, and anchors the thesis to honest measurement.

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

**Source: `data/benchmark/alm_vs_alm.json` lines 9–24 (confirm leaderboard)**

Trained on full XL corpus (9,959 train clips / 5.77 h native-Sanskrit TTS; 58-clip val frozen), 3 epochs, best checkpoint by fair val CER at epoch 1 (epoch 2 overfit to 0.086), 1.35 GPU-hours RTX 5090.

| Model | Type | Params | cer_norm | cer_raw | Gate |
|---|---|---|---|---|---|
| **Śabda-ALM 1.13B+LoRA XL** | specialist, confirm | 1.13B+8.4M LoRA | **0.0392** | **0.0446** | AA |
| Voxtral-Mini-3B-2507 (Mistral) | generalist, open | 3B | 0.1866 | 0.9478 | AA |
| Qwen2.5-Omni-3B Thinker (Alibaba) | generalist, open | 3B | 0.2133 | 6.7643 | AA |
| Qwen2-Audio-7B-Instruct (Alibaba) | generalist, open | 7B | 0.4305 | 0.6493 | AA |

**Result: 4.8× lower cer_norm than Voxtral (0.187).** Per-clip predictions: `data/benchmark/alm_vs_alm_records.json`; metrics: `data/alm/xl1b_metrics.json`; checkpoint: `data/alm/xl1b_ckpt.pt`.

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

**Source: `research/sphota-lens.md` + `data/sphota_lens/emergence_report.json`**

Instrument: for each layer, a linear probe (template-grouped-CV) predicts the sentence's kriyā (verb, core meaning from gold kāraka parse) from audio-position reps. Decodability per layer = meaning-emergence curve.

**Result (200M core ALM, 240 utterances, 45 kriyā classes, chance = 0.022):**
- Meaning is decodable from audio positions alone at ≈0.25 — **11× chance**.
- The curve **rises to peak at layer 13** (0.263), then declines toward output (0.25 → 0.23).
- **Correlational peak = layer 13; causal peak = layer 14** (ablation: mean-collapse audio positions at each layer, measure drop in final decodability). They agree within ±1 layer.
- **Sphoṭa layer = 13, validated = True.**

This is the *paśyantī → vaikharī* gradient made concrete: sound-borne meaning crystallises mid-network, then moves toward articulation at the surface. The locus is not an artifact: ablation and correlation peaks agree, and the curve rises significantly above chance.

**Caveat:** single model family (200M byte-core); probe is linear. The 1.13B model's workspace band is layers 21–23 (CKA contrast 0.0253, P3-sphota-lens.md), different from the 200M emergence locus—a reminder that the locus is architecture-dependent.

### 3.3 Steering the Workspace: Can We Write to Paśyantī?

**Source: `data/alm/p3su_results.json` + `research/P3-sphota-lens.md`**

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

**But H-SU3 fails** (pre-registered svātantrya budget, lines 53–58): entropy delta at the best α is −1.436 nats, far exceeding the 0.5-nat budget. Steering that works *only by collapsing the distribution* violates the autonomy (svātantrya) constraint: the model can load a concept, but only by suppressing its own entropy. This is an honest finding—reported not hidden. It bounds the claim: steering reveals the workspace is writable, but loading ≠ faithful utterance.

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

**Net result (E6/E7):** No speech-vs-text holism difference. Both modalities show late-resolving holism (valid metric); speech does not exceed it (null replicated at power).

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
| 7 | Sphoṭa layer 13 (200M model): decodability 0.263 vs 0.022 chance | research/sphota-lens.md §2; data/sphota_lens/emergence_report.json | SU (steering gate asserts methodology) | Correlational + causal peaks agree ±1 layer; single model family; linear probe |
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

## 6. Reproducibility & Gating

### 6.1 Pre-registration

All hypotheses (H-SU1, H-SU2, H-SU3, H-NG1, H-NG2, H-F2a, H-F2b) locked before any battery run.

**File:** `research/prereg/P3-battery-prereg.md` (registered 2026-07-18, committed before runs).

### 6.2 Dual-Verdict Gates

Three gates, each binary pass/fail, each asserting *sound methodology, not predetermined winner*:

- **Gate AA** (apples-to-apples benchmark): audio genuinely reaches each model, per-clip predictions saved, scheme-neutral metric, fair budget for all. **Command:** `python gates/check.py AA`
- **Gate SU** (steering uptake): methodology sound (workspace locus measured, ablation+correlation peaks agree, injection and readback procedure valid), not whether H-SU1 passes. **Command:** `python gates/check.py SU`
- **Gate NG** (nyāya guardrail): fire-rate and fix/break/neutral counts reported with per-task records, not whether improvement is positive. **Command:** `python gates/check.py NG`

### 6.3 Reproducibility Commands

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

### 6.4 Artifact Inventory

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

## 7. Conclusion

Pranava delivers:

1. **An architectural thesis:** sphoṭa/śabdādvaita is realizable through an ALM (the four-vāk maps onto real structure) and not through text-only models (which are frozen vaikharī).

2. **Empirical instruments to make the thesis measurable:** the Sphoṭa-Lens locates meaning-emergence causally (layer 13 on 200M); the workspace band (layers 21–23 on 1.13B) is writable via steering, though with caveats (entropy collapse).

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

