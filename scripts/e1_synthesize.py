"""E1: synthesize the controlled stimulus set to WAV with a single fixed voice.

Deterministic neural TTS (piper, en_US-lessac-medium) → consistent speaker so that
acoustic variation across items reflects the *content manipulation*, not voice. Writes
data/stimuli/manifest.jsonl (text, labels, wav path, duration) + a datasheet.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

from pranava.experiments.stimuli import generate_stimuli

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / ".voices" / "en_US-lessac-medium.onnx"
OUT = ROOT / "data" / "stimuli"
WAV_DIR = OUT / "wav"


def main() -> int:
    from piper import PiperVoice

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(str(VOICE))
    stimuli = generate_stimuli()

    rows = []
    for st in stimuli:
        wav_path = WAV_DIR / f"{st.id}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            voice.synthesize_wav(st.text, wf)
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
        rows.append(
            {
                **st.to_dict(),
                "wav": str(wav_path.relative_to(ROOT)),
                "sr": sr,
                "duration_s": round(nframes / sr, 4),
            }
        )

    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    from collections import Counter

    res = Counter(r["resolution"] for r in rows)
    struct = Counter(r["structure"] for r in rows)
    datasheet = {
        "n_items": len(rows),
        "voice": VOICE.name,
        "sample_rate": rows[0]["sr"] if rows else None,
        "resolution_balance": dict(res),
        "structure_balance": dict(struct),
        "total_audio_s": round(sum(r["duration_s"] for r in rows), 1),
        "generation": "template slot-fill; labels+disambig index exact by construction",
        "limits": "synthetic single-speaker TTS; English; small closed vocabulary; "
        "disambiguation index is word-level (frame alignment approximated in E2).",
    }
    (OUT / "datasheet.json").write_text(json.dumps(datasheet, indent=2), encoding="utf-8")
    print(json.dumps(datasheet, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
