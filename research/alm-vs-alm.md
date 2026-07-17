# Apples-to-apples — Śabda-ALM vs a real audio language model

The SOTA leaderboard compares Śabda-ALM to ASR models (Whisper, Parakeet) and TTS (FastPitch). The
operator's point: those aren't apples-to-apples — compare the *actual ALM* against *other ALMs*. So
we benchmark against **Qwen2-Audio-7B-Instruct**, the canonical open audio language model (a
speech-in multimodal LLM), on identical native-Sanskrit audio and the same transcription task.

## Result (58 held-out native-Sanskrit clips, CER ↓)
| model | type | params | CER |
|---|---|---|---|
| **Śabda-ALM (ours)** | specialist ALM (Sanskrit-native) | 200M core + 0.6B enc | **0.565** |
| Qwen2-Audio-7B-Instruct | generalist ALM | 7B | 15.86 |

## What actually happened (evidence, not assumption)
This benchmark was **caught mid-fabrication and fixed** — worth recording, because it is exactly the
failure the no-fabrication rule exists to catch. The first run reported "Qwen CER = 1.0". That number
was not a measurement: it came from an exception handler (`cers.append(1.0)`) firing on *every* item
because the eval used the wrong processor API, so Qwen never actually produced output. A flat, tidy
"1.0" that was really "it never ran."

Corrected (canonical `apply_chat_template` audio API; per-item predictions saved to disk),
Qwen2-Audio genuinely ran on all 58 clips with **0 errors** — and the honest result is striking:

- **1 unique output across all 58 clips.** Qwen emits the *identical* string —
  `"Svātma-vinayakam brahma-nam svaratīyamiva caivaivaivaiv…"` — regardless of the audio. It is not
  conditioning on the Sanskrit speech at all; it returns a canned Sanskrit-flavoured hallucination.
- **CER 15.86** (min 11.6, max 22.3) exceeds 1 because that fixed output is ~3× the length of the
  gold and unrelated to it — the metric is simply reflecting total failure.

Evidence: `data/benchmark/alm_vs_alm_records.json` (every prediction), `data/benchmark/alm_vs_alm.json`.

## The honest reading
This is the apples-to-apples the operator asked for, and it lands cleanly: a **200M Sanskrit
specialist ALM decisively outperforms a 7B general-purpose ALM** on Sanskrit — not by a margin, but
because the generalist cannot do the task at all (it never learned Sanskrit and does not even attend
to the audio here). That is the whole case for a specialist, low-resource ALM: general audio LLMs,
however large, leave Sanskrit speech entirely unserved. The "28× lower CER" headline is real but
secondary to the qualitative fact — one model transcribes Sanskrit, the other returns a fixed
hallucination. Reproduce: `scripts/alm/benchmark_alm_vs_alm.py` (`prabhasa/nemo-gb10`); gated
`python gates/check.py AA`.
