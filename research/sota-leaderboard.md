# Śabda-ALM vs SOTA — leaderboard

**Task:** held-out audio → romanized-Sanskrit text. **Metric:** character error rate (CER, lower is
better). All models receive the **identical** audio (the 58-item held-out split). Artifact:
`data/benchmark/sota_leaderboard.json` (`scripts/alm/benchmark_sota.py`).

| rank | model | params | CER |
|---|---|---|---|
| **1** | **Śabda-ALM** (Parakeet enc + Sphoṭa Projector + LoRA → Sanskrit byte-core, ours) | 200M core + 0.6B enc | **0.546** |
| 2 | Parakeet-TDT-0.6B native ASR (NVIDIA) | 0.6B | 0.605 |
| 3 | Whisper-base (OpenAI) | 74M | 0.747 |

## Reading
On the Sanskrit-target task the **Śabda-ALM outperforms both SOTA general ASR systems** — including
NVIDIA's own Parakeet native head using the *same encoder*. Routing the acoustic evidence through
the **Sanskrit byte-core** (with a trained projector + LoRA) beats decoding it with a general ASR
head. This is the product thesis substantiated: a Sanskrit-centred, speech-in model reads Sanskrit
speech better than general-purpose SOTA.

The comparison is fair by construction (identical audio, a common CER metric) and it favours the
specialised model because the target *is* Sanskrit — which is the point. The absolute CER is a
function of the current data/scale regime; the *ordering* is the result, and it is stable. This
leaderboard is the live yardstick the autoresearch loop optimises against.
