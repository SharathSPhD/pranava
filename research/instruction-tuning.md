# Instruction-tuning the Śabda-ALM — one audio clip, six tasks

The multilingual ALM only transcribed. Instruction-tuning makes it a usable, *promptable* model:
the same audio clip yields different structured answers depending on a short English instruction —
transcribe it, name its language, or extract a specific kāraka (kartā / karaṇa / karma / kriyā).

## Method
- **Data (gold-grounded, nothing fabricated).** 3,947 (audio, instruction, response) examples built
  from labels already on disk — the native-Sanskrit corpus's gold kāraka parse and the language tag.
  Six tasks; a task is emitted for a clip only when its gold answer exists. `instruct_corpus/`.
- **Architecture.** The core already accepts an embedding prefix (the projected audio). The
  instruction is embedded through the core's own byte-embedding and concatenated *after* the audio:
  `[audio tokens] ++ [instruction bytes] → generate [answer] + EOS`. Warm-started from the
  multilingual checkpoint, then projector + LoRA are SFT'd with a response-only loss.
- **EOS.** Responses are trained with an end-of-answer sentinel (byte 0) and decoding halts at it —
  without this the byte decoder runs to `max_new` and rambles, and *nothing* exact-matches.

## Result (held-out, 381 examples) — SFT
| task | exact-match accuracy |
|---|---|
| karma (object) | **0.868** |
| kartā (agent) | **0.759** |
| kriyā (action) | 0.466 |
| karaṇa (means) | 0.091 |
| transcribe | 0.019 |
| language | 0.000 |
| **overall** | **0.281** |

**The model genuinely follows the instruction — a clean causal test.** With the *correct*
instruction, overall accuracy is 0.281; with a **shuffled** instruction on the same audio it is
**exactly 0.000**. The instruction-sensitivity gap (0.281) is the whole of the accuracy: the model
answers the question it is asked, not a fixed one. Gated: `python gates/check.py IT`.

## The honest reading — where it wins and where it doesn't
- **Extractive single-word tasks are strong** (karma 0.87, kartā 0.76): these targets are
  in-distribution Sanskrit words the byte-core emits cleanly. karaṇa is weak (0.09) — the rarest
  role, only 125 training examples.
- **language = 0.00 is a real, legible limitation.** Every prediction *starts* "Sans…" — the model
  has learned that a language question about Sanskrit should begin that way — but its byte-prior is
  overwhelmingly Sanskrit, so it cannot cleanly complete the English meta-label "Sanskrit" and
  drifts back into Sanskrit text ("SansitAm", "Sanskrispane"). The model can *speak* Sanskrit far
  more easily than it can *name* it in English.
- **transcribe = 0.02 exact-match understates it.** Predictions are recognisably the right
  utterance with morphological slips ("vidye vane BavizyataH" → "vidyO vanam … vadizyataH"); exact
  string match is unforgiving on long sequences. CER (≈0.7, see the multilingual benchmark) is the
  fairer measure for this task.

Both weak spots — the run-on drift on `language` and the near-miss rambling on `transcribe` — are
what the RLAIF pass targets with an AI-feedback reward that prizes correctness **and** conciseness
(`research/rlaif.md`). Artifacts: `data/alm/instruct_metrics.json`, `data/alm/instruct_ckpt.pt`.
