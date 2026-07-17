# Śabda-ALM vs SOTA — the honest leaderboard (two audio regimes)

**Task:** held-out audio → romanized-Sanskrit text; **metric:** character error rate (CER, lower is
better); identical audio for every model. We test on **two** audio distributions, and the result
differs — which is itself the finding.

### On authentic native-Sanskrit speech (AI4Bharat indic-parler-tts, Devanāgarī phonetics)
| rank | model | params | CER |
|---|---|---|---|
| 1 | Parakeet-TDT-0.6B native ASR (NVIDIA) | 0.6B | **0.522** |
| **2** | **Śabda-ALM** (Parakeet + Projector + LoRA → Sanskrit core, ours) | 200M core | **0.565** |
| 3 | Whisper-base (OpenAI) | 74M | 0.754 |

### On a controlled English-FastPitch TTS distribution
| rank | model | params | CER |
|---|---|---|---|
| **1** | **Śabda-ALM** (ours) | 200M core | **0.546** |
| 2 | Parakeet-TDT-0.6B native ASR (NVIDIA) | 0.6B | 0.605 |
| 3 | Whisper-base (OpenAI) | 74M | 0.747 |

## The honest reading
The ALM's first-place finish on the FastPitch corpus was **partly specific to that TTS
distribution**: the model trained on it, while general ASR handles the English-voiced romanized
Sanskrit poorly. On **authentic Sanskrit phonetics**, NVIDIA's multilingual Parakeet-TDT-v3 — whose
training covers Indic languages — transcribes the clearer native pronunciation better (0.522), and
the small 200M Śabda-ALM is a **close second (0.565)**, still well ahead of Whisper.

So the defensible claim is not "beats SOTA" but the sharper, true one: **a 200M speech model on a
Sanskrit byte-core is competitive with a 0.6B SOTA multilingual ASR on authentic Sanskrit speech**,
and the $0.043$ CER gap is a concrete target. This is why we built the harder test: the controlled
corpus flattered the specialist; the authentic corpus is the real yardstick, and the loop now
optimises against it. Next iterations to close the gap: LoRA on the 1.13B core, more epochs, and a
larger corpus — all queued in the EFE loop.

Artifacts: `data/benchmark/sota_leaderboard.json` (native) and `..._fastpitch.json` (controlled);
`scripts/alm/benchmark_sota.py`.
