"""Phase 1 — build a paired (audio, Sanskrit text, kāraka) corpus for ALM training.

Sentences come from PSALM's Vidyut-realized fixture (`paninian_v1.jsonl` — romanized Sanskrit with
gold kāraka *by construction*, the ADR-0033 replacement for the broken Saṃsādhanī). Audio is
synthesized with NeMo FastPitch+HiFiGAN. Writes a manifest + datasheet + a fixed train/val split.

HONEST LIMITATION (datasheet): the TTS is an English FastPitch voice pronouncing romanized Sanskrit
— consistent and deterministic (good for learning a speech→text projector) but NOT native Sanskrit
phonetics. Native-pronunciation audio is future work. Run in prabhasa/nemo-gb10 via in_container.sh.
"""
from __future__ import annotations

import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
FIXTURE = Path("/work/PSALM/data/fixtures/paninian_v1.jsonl")
if not FIXTURE.exists():
    FIXTURE = Path("/home/sharaths/projects/PSALM/data/fixtures/paninian_v1.jsonl")
OUT = ROOT / "data" / "alm" / "speech_corpus"
WAV_DIR = OUT / "wav"
SR = 22050


def load_sentences(n: int) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for line in FIXTURE.open(encoding="utf-8"):
        r = json.loads(line)
        text = r.get("text", "").strip()
        # keep short, clean, unique sentences (TTS-friendly)
        if not text or text in seen or not (8 <= len(text) <= 80):
            continue
        seen.add(text)
        rows.append({"text": text, "karaka": r.get("karaka_parse", []), "source": "paninian_v1"})
        if len(rows) >= n:
            break
    return rows


def split_of(text: str) -> str:
    h = int(hashlib.sha1(text.encode()).hexdigest(), 16)
    return "val" if (h % 10 == 0) else "train"


def main(n: int = 600) -> int:
    import torch
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    sents = load_sentences(n)
    if not sents:
        print("no sentences loaded", file=sys.stderr)
        return 2

    sp = FastPitchModel.from_pretrained("tts_en_fastpitch").eval().cuda()
    voc = HifiGanModel.from_pretrained("tts_en_lj_hifigan_ft_mixertts").eval().cuda()

    rows = []
    total_samples = 0
    for i, s in enumerate(sents):
        wav_path = WAV_DIR / f"u{i:05d}.wav"
        with torch.no_grad():
            toks = sp.parse(s["text"])
            spec = sp.generate_spectrogram(tokens=toks)
            audio = voc.convert_spectrogram_to_audio(spec=spec)[0].cpu().numpy().astype(np.float32)
        # write 16-bit PCM
        pcm = np.clip(audio, -1, 1)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
            wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
        total_samples += len(audio)
        rows.append({**s, "id": f"u{i:05d}", "wav": str(wav_path.relative_to(ROOT)),
                     "sr": SR, "duration_s": round(len(audio) / SR, 3), "split": split_of(s["text"])})

    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    from collections import Counter
    sp_ct = Counter(r["split"] for r in rows)
    datasheet = {
        "n_items": len(rows), "total_hours": round(total_samples / SR / 3600, 3),
        "sample_rate": SR, "source": "PSALM paninian_v1 (Vidyut-realized, gold kāraka)",
        "tts": "NeMo tts_en_fastpitch + tts_en_lj_hifigan_ft_mixertts",
        "split": dict(sp_ct),
        "limitation": "English FastPitch voice pronouncing romanized Sanskrit — consistent/"
        "deterministic but not native Sanskrit phonetics; native-pronunciation audio is future work.",
        "text_encoding": "SLP1-romanized; gold kāraka by construction (no parser).",
    }
    (OUT / "datasheet.json").write_text(json.dumps(datasheet, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(datasheet, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 600))
