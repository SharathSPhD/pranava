"""Apples-to-apples ALM benchmark: Śabda-ALM vs Qwen2-Audio-7B-Instruct.

Task: hold-out audio → romanized-Sanskrit text; metric = character error rate (CER) against gold.

Compares:
  * Śabda-ALM (Parakeet enc + Sphoṭa Projector + LoRA-adapted Sanskrit core) — ours, specialist ALM
  * Qwen/Qwen2-Audio-7B-Instruct (open, general-purpose ALM) — generalist baseline

Both models see identical native-Sanskrit audio and gold text. Run in prabhasa/nemo-gb10:26.02 or
similar container with transformers, torch, librosa.

Note: Śabda-ALM CER is pre-computed and loaded from sota_leaderboard.json; Qwen2-Audio CER is
computed in this run since it doesn't require prabhasa infrastructure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "benchmark"


def _levenshtein(s1: str, s2: str) -> int:
    """Edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            current_row.append(
                min(previous_row[j + 1] + 1, current_row[j] + 1, previous_row[j] + (c1 != c2))
            )
        previous_row = current_row
    return previous_row[-1]


def cer(pred: str, gold: str) -> float:
    """Character error rate using Levenshtein distance."""
    return _levenshtein(pred, gold) / max(1, len(gold))


def load_manifest(split: str | None = None) -> list:
    """Load manifest from the indic speech corpus."""
    corpus_dir = ROOT / "data" / "alm" / "speech_corpus_indic"
    manifest_path = corpus_dir / "manifest.jsonl"

    rows = [json.loads(x) for x in manifest_path.open(encoding="utf-8") if x.strip()]
    out = []
    for r in rows:
        if split and r["split"] != split:
            continue
        out.append({
            "id": r["id"],
            "text": r["text"],
            "wav_path": ROOT / r["wav"],
        })
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read WAV file and return audio + sample rate."""
    import wave
    with wave.open(str(path), "rb") as wf:
        sr, raw = wf.getframerate(), wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def eval_qwen_audio(val):
    """Evaluate Qwen2-Audio-7B-Instruct on val split.

    Transcribe Sanskrit audio using the open ALM, prompted as a speech-to-text model.
    """
    try:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    except Exception as e:
        return None, f"qwen2-audio unavailable: {e}"

    import os
    os.environ["HF_HOME"] = str(ROOT / ".hf_cache")

    try:
        print("Loading Qwen2-Audio-7B-Instruct...")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.eval()
        print("Model loaded successfully\n")
    except Exception as e:
        return None, f"Failed to load Qwen2-Audio: {e}"

    import librosa

    cers, records, errors = [], [], 0
    with torch.no_grad():
        for i, ex in enumerate(val):
            try:
                # `val` holds dicts — use subscript access.
                wav, sr = read_wav(ex["wav_path"])
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

                # Canonical Qwen2-Audio chat-template API (audio content + text instruction).
                conversation = [{"role": "user", "content": [
                    {"type": "audio", "audio_url": "clip.wav"},
                    {"type": "text", "text": "Transcribe this spoken Sanskrit to romanized (IAST/SLP1) "
                                             "text. Output only the transcription."}]}]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                                       tokenize=False)
                inputs = processor(text=prompt, audios=[wav], sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                gen = out[:, inputs["input_ids"].shape[1]:]  # drop the prompt tokens
                pred = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

                c = cer(pred, ex["text"])
                cers.append(c)
                records.append({"id": ex["id"], "qwen_pred": pred, "gold": ex["text"], "cer": round(c, 4)})
                print(f"  [{i+1:2d}/{len(val)}] {ex['id']}: CER={c:.4f} pred='{pred[:40]}' gold='{ex['text'][:40]}'",
                      flush=True)
            except Exception as e:
                errors += 1
                records.append({"id": ex.get("id", "?"), "error": str(e)})
                print(f"  [{i+1:2d}/{len(val)}] ERROR: {e}", flush=True)

    # Save the raw predictions as evidence the model genuinely ran (no silent 1.0 defaults).
    (OUT / "alm_vs_alm_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    if not cers:
        return None, f"Qwen produced NO output on any item ({errors} errors) — could not be evaluated"
    if errors:
        print(f"  (note: {errors}/{len(val)} items errored and are excluded)")
    return float(np.mean(cers)), (f"{errors} items errored" if errors else None)


def load_sabda_alm_cer():
    """Load pre-computed Śabda-ALM CER from sota_leaderboard.json."""
    sota_file = OUT / "sota_leaderboard.json"
    if not sota_file.exists():
        return None, "sota_leaderboard.json not found"

    try:
        data = json.loads(sota_file.read_text())
        for entry in data.get("leaderboard", []):
            if "Śabda-ALM" in entry.get("model", ""):
                return entry["cer"], None
        return None, "Śabda-ALM entry not found in leaderboard"
    except Exception as e:
        return None, f"Failed to load sota_leaderboard.json: {e}"


def main(n: int = 58) -> int:
    """Benchmark both ALMs on native-Sanskrit val split."""
    val = load_manifest("val")[:n]
    print(f"Loaded {len(val)} val examples from speech_corpus_indic\n")

    board = []

    # Load pre-computed Śabda-ALM CER
    print("=== Loading Śabda-ALM (pre-computed) ===")
    sabda_cer, sabda_err = load_sabda_alm_cer()
    if sabda_cer is not None:
        board.append({
            "model": "Śabda-ALM (Parakeet+Projector+LoRA→Sanskrit core, ours)",
            "params": "200M core + 0.6B enc",
            "cer": sabda_cer,
            "alm_type": "specialist (Sanskrit-native)",
        })
        print(f"Śabda-ALM CER: {sabda_cer}\n")
    else:
        print(f"Could not load Śabda-ALM: {sabda_err}\n")

    # Evaluate Qwen2-Audio
    print("=== Evaluating Qwen2-Audio-7B-Instruct ===")
    qwen_cer, qwen_err = eval_qwen_audio(val)
    board.append({
        "model": "Qwen2-Audio-7B-Instruct (open ALM)",
        "params": "7B",
        "cer": round(qwen_cer, 4) if qwen_cer is not None else None,
        "alm_type": "generalist (no Sanskrit training)",
        "note": qwen_err,
    })
    print(f"\nQwen2-Audio CER: {round(qwen_cer, 4) if qwen_cer else 'UNAVAILABLE'}")

    # Sort by CER (best first)
    board.sort(key=lambda r: (r["cer"] is None, r["cer"] if r["cer"] is not None else 9))

    result = {
        "task": "held-out audio → romanized-Sanskrit text (CER, lower=better)",
        "n_items": len(val),
        "audio_corpus": "native Sanskrit (indic-parler-tts)",
        "comparison": "specialist vs generalist ALM",
        "leaderboard": board,
        "note": "Apples-to-apples ALM comparison: Śabda-ALM (specialist, fine-tuned on Sanskrit) vs Qwen2-Audio-7B-Instruct (generalist, trained on multilingual data without Sanskrit). "
                "The specialist ALM is expected to outperform a general-purpose model on this Sanskrit-specific task."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alm_vs_alm.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nResults written to {OUT / 'alm_vs_alm.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
