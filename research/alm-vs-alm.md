# Apples-to-apples — Śabda-ALM vs open audio language models

## Correction (2026-07-18) — the earlier result here was wrong
The previously committed conclusion — *"the 200M specialist Śabda-ALM (CER 0.565) decisively beats 7B
Qwen2-Audio (CER 15.86), 28×, because the generalist can't do Sanskrit and ignores the audio"* — does
**not** survive a correct harness. It rested on two bugs, one in each direction; fixed, the conclusion
reverses.

### Bug 1 — Qwen never received the audio
The eval called `processor(text=prompt, audios=[wav], ...)`. In transformers 4.57 the Qwen2-Audio
processor's audio kwarg is **`audio`** (singular); **`audios` is silently ignored** (a warning, not an
error), so no `input_features` were ever built and Qwen generated from the text prompt alone — one
identical canned string (`"Svātma-vinayakam …caivaivaiva…"`) for all 58 clips, CER 15.86. The prior
write-up read that as "Qwen ignores the audio"; in fact the harness never fed it any.
Fix: `audio=[wav]`. Verified the audio now reaches the model — `input_features` present, per-clip
outputs vary (57–58 unique of 58), and differ from a zero-audio control.

### Bug 2 — the specialist was handed the answer length
The specialist's 0.565 came from decoding `len(gold)+4` bytes and truncating to `len(gold)` — a
**gold-length oracle** the generalists never get. The specialist has no reliable EOS, so under a fair
fixed budget it runs to the full length and appends repetition. Remove the oracle → free-decode CER
1.82. We keep the oracle number as a clearly-labeled "capped" variant for continuity only.

## Method
- 58 held-out native-Sanskrit clips (indic-parler-tts) → romanized text; metric = character error rate.
- **Primary = normalized CER**: every output and the gold are folded to an ASCII phonetic skeleton
  (SLP1 / Devanagari → IAST via indic-transliteration, diacritics stripped, lowercased, leading language
  tags and chat continuations removed). Scheme-neutral, so a Devanagari transcription and an SLP1 one are
  compared on **phonetics, not spelling** — the only fair way to score models that answer in different
  scripts. Secondary = raw CER vs the SLP1 gold (favours the SLP1-native specialist).
- Decode: greedy, fixed 64-token budget, **no gold-length oracle** (except the labeled capped variant).
- Models: our 200M specialist (free + capped) vs open generalist ALMs **Qwen2-Audio-7B**,
  **Qwen2.5-Omni-3B**, **Voxtral-Mini-3B**. (Ultravox-1B was excluded: its weights are gated behind
  Llama-3.2 access this account lacks — documented, not silently dropped.)

## Result (normalized CER ↓ is the primary, fair metric)
| model | type | params | cer_norm | cer_raw |
|---|---|---|---|---|
| **Voxtral-Mini-3B-2507** (Mistral) | generalist | 3B | **0.187** | 0.948 |
| **Qwen2.5-Omni-3B Thinker** (Alibaba) | generalist | 3B | **0.213** | 6.764\* |
| Qwen2-Audio-7B-Instruct (Alibaba) | generalist | 7B | 0.431 | 0.649 |
| Śabda-ALM — capped / gold-length oracle (ours) | specialist+oracle | 200M+0.6B enc | 0.506 | 0.748 |
| Śabda-ALM — free decode (ours) | specialist | 200M+0.6B enc | 1.819 | 2.158 |

\* Omni's *raw* CER is high because it transcribes correctly and then appends a hallucinated
`"Human: What is the meaning of …"` dialogue; the normalized metric scores the transcription line only
(uniform cleanup for all models).

Per-clip examples (gold → model output):
- `naraH gfham paWet` → Voxtral `नराग्रहं पठेत्` · Omni `Nara griham pathe` · Qwen2-Audio `Nara-Griham-Patate` — all essentially correct.
- same clip → Śabda-ALM (free) `[sa] naraH gfeean…gacCanti…` — right start, then cannot stop.

## The honest reading
Wired correctly and scored on a scheme-neutral metric, **open generalist ALMs transcribe this Sanskrit
better than the 200M specialist.** Voxtral-3B (0.19) and Qwen2.5-Omni-3B (0.21) lead, Qwen2-Audio-7B
(0.43) follows; the specialist reaches 0.51 only *with* a gold-length oracle and 1.82 without. The
earlier "specialists are the whole case; general audio LLMs leave Sanskrit unserved" claim was an
artifact — the generalists were simply never handed the audio, and the specialist was handed the answer
length.

What remains true: the specialist is 15–35× smaller and emits SLP1 directly (so its *raw* CER 0.75 beats
Voxtral's raw 0.95, which answers in Devanagari) — but on transcription accuracy, measured fairly, it
does not lead. That is an honest negative, and it is the useful result: the case for the specialist has
to be made on grounds other than beating these models at transcription (e.g. size, the kāraka/structured
tasks, latency, on-device use), not on a transcription win that only existed because of a broken harness.

Reproduce: `scripts/alm/benchmark_alm_vs_alm.py` (prabhasa/nemo-gb10) → raw per-clip predictions, then
`scripts/alm/_rescore_alm.py` (host venv) → the normalized leaderboard. Evidence:
`data/benchmark/alm_vs_alm{,_records}.json`. Gated: `python gates/check.py AA` — the gate asserts sound
methodology (audio genuinely reaches each model, per-clip predictions saved, a fair normalized metric),
**not** a predetermined winner.
