"""Build a NATIVE-Sanskrit speech corpus with AI4Bharat indic-parler-tts.

Real human Sanskrit speech is unavailable (FLEURS lacks Sanskrit; Common Voice is gated/sparse), so
the most authentic option is a proper Indic TTS with native Sanskrit phonetics — a real upgrade over
the English-FastPitch-on-romanized-Sanskrit corpus. Reuses the same sentences / labels / splits as
the original corpus; only the audio is re-synthesized (SLP1 → Devanāgarī → native Sanskrit speech).
Runs on the host GPU (parler_tts; no mamba needed). Loads the model from its local cache snapshot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from indic_transliteration import sanscript

ROOT = Path("/home/sharaths/projects/pranava")
SRC_MANIFEST = ROOT / "data/alm/speech_corpus/manifest.jsonl"
OUT = ROOT / "data/alm/speech_corpus_indic"
WAV = OUT / "wav"
SNAP = next(Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--ai4bharat--indic-parler-tts/snapshots")).glob("*/"))
DESC = "A clear male voice speaks slowly and distinctly with a neutral tone in a quiet room."


def main(limit: int | None = None) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(str(SNAP)).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(str(SNAP))
    sr = int(model.config.sampling_rate)
    WAV.mkdir(parents=True, exist_ok=True)

    rows_in = [json.loads(x) for x in SRC_MANIFEST.open(encoding="utf-8") if x.strip()]
    if limit:
        rows_in = rows_in[:limit]
    desc_ids = tok(DESC, return_tensors="pt").to("cuda")

    rows, total = [], 0
    for r in rows_in:
        dev_text = sanscript.transliterate(r["text"], sanscript.SLP1, sanscript.DEVANAGARI)
        p = tok(dev_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            aud = model.generate(input_ids=desc_ids.input_ids,
                                 attention_mask=desc_ids.attention_mask,
                                 prompt_input_ids=p.input_ids,
                                 prompt_attention_mask=p.attention_mask)
        a = aud.cpu().numpy().squeeze().astype(np.float32)
        wp = WAV / f"{r['id']}.wav"
        sf.write(str(wp), a, sr)
        total += len(a)
        rows.append({**r, "text_devanagari": dev_text, "wav": str(wp.relative_to(ROOT)),
                     "sr": sr, "duration_s": round(len(a) / sr, 3)})
        if len(rows) % 50 == 0:
            print(json.dumps({"done": len(rows), "of": len(rows_in)}), flush=True)

    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    ds = {"n_items": len(rows), "total_hours": round(total / sr / 3600, 3), "sample_rate": sr,
          "tts": "ai4bharat/indic-parler-tts (NATIVE Sanskrit phonetics, Devanāgarī input)",
          "provenance": "same sentences/labels/splits as speech_corpus; audio re-synthesized native",
          "note": "real human Sanskrit speech unavailable (FLEURS no sa; Common Voice gated) — this is "
                  "the most authentic synthesis available."}
    (OUT / "datasheet.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ds, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
