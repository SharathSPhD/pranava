"""Build the BILINGUAL instruction corpus — the substance behind the chat (contract clause 2).

Tasks assembled (all grounded, nothing fabricated):
  * translate sa→en : audio = native-Sanskrit TTS of the Itihāsa Sanskrit verse; instruction
                      "translate to english"; response = the ALIGNED Itihāsa English line (lowercased).
  * translate en→sa : audio = TTS of the English line; instruction "translate to sanskrit";
                      response = the aligned Sanskrit in SLP1.
  * language-ID     : reuse the same clips; instruction "which language is this";
                      response "Sanskrit"/"English".
Transcribe (real audio, both languages) and kāraka items come from the existing public/synthetic
corpora at training time — this builder only creates what does not yet exist (the translation audio).

Itihāsa: aligned CSVs at /home/sharaths/projects/slm-1/data/itihasa (train.{en,sn}.csv). We sample
SHORT pairs (TTS-friendly, ≤140 chars both sides) from the TRAIN files only. TTS: ai4bharat
indic-parler-tts — Devanagari input for Sanskrit (native phonetics), English text for English.

Writes data/alm/bi_instruct/{wav/, manifest.jsonl, datasheet.json}. GPU (host venv):
  .venv/bin/python scripts/alm/build_bi_instruct.py --n 1500
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ITI = Path("/home/sharaths/projects/slm-1/data/itihasa")
OUT = ROOT / "data/alm/bi_instruct"
WAV = OUT / "wav"

DESC_SA = "A clear male voice speaks slowly and distinctly with a neutral tone in a quiet room."
DESC_EN = "A clear male voice speaks English slowly and distinctly with a neutral tone in a quiet room."
_DEVA = re.compile(r"[ऀ-ॿ]")


def load_pairs(limit_chars: int = 140) -> list[tuple[str, str]]:
    en = [r[0] if r else "" for r in csv.reader((ITI / "train.en.csv").open(encoding="utf-8"))]
    sn = [r[0] if r else "" for r in csv.reader((ITI / "train.sn.csv").open(encoding="utf-8"))]
    pairs = []
    for e, s in zip(en, sn):
        e, s = e.strip(), s.strip()
        if not e or not s or len(e) > limit_chars or len(s) > limit_chars:
            continue
        if not _DEVA.search(s):  # sanskrit side must be Devanagari
            continue
        if not re.search(r"[a-zA-Z]", e):
            continue
        pairs.append((e, s))
    return pairs


def main(n: int = 1500) -> int:
    import soundfile as sf
    import torch
    from indic_transliteration import sanscript

    snaps = Path.home() / ".cache/huggingface/hub/models--ai4bharat--indic-parler-tts/snapshots"
    local = next(snaps.glob("*/"), None)
    src = str(local) if local else "ai4bharat/indic-parler-tts"
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(src).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(src)
    sr = int(model.config.sampling_rate)
    WAV.mkdir(parents=True, exist_ok=True)
    d_sa = tok(DESC_SA, return_tensors="pt").to("cuda")
    d_en = tok(DESC_EN, return_tensors="pt").to("cuda")

    pairs = load_pairs()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))[:n]
    manifest = OUT / "manifest.jsonl"
    done_ids = set()
    if manifest.exists():  # resumable
        done_ids = {json.loads(l)["id"] for l in manifest.open(encoding="utf-8") if l.strip()}
    out_f = manifest.open("a", encoding="utf-8")

    def tts(text: str, desc, cid: str) -> dict | None:
        wp = WAV / f"{cid}.wav"
        if not wp.exists():
            p = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                aud = model.generate(input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                                     prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            a = np.atleast_1d(aud.cpu().numpy().squeeze().astype(np.float32))
            if a.ndim != 1 or len(a) < int(0.3 * sr):
                return None
            sf.write(str(wp), a, sr)
        return {"wav": str(wp.relative_to(ROOT)), "sr": sr}

    made = 0
    for k, i in enumerate(idx):
        e, s = pairs[int(i)]
        slp1 = sanscript.transliterate(s, sanscript.DEVANAGARI, sanscript.SLP1)
        base = f"bi_{int(i):06d}"
        if f"{base}_sa" not in done_ids:
            w = tts(s, d_sa, f"{base}_sa")  # native Devanagari input
            if w:
                for instr, resp, task in (("translate to english", e.lower(), "translate_en"),
                                          ("which language is this", "Sanskrit", "language")):
                    out_f.write(json.dumps({"id": f"{base}_sa", "task": task, "lang": "sa",
                                            "instruction": instr, "response": resp,
                                            "source_text": slp1, **w}, ensure_ascii=False) + "\n")
        if f"{base}_en" not in done_ids:
            w = tts(e, d_en, f"{base}_en")
            if w:
                for instr, resp, task in (("translate to sanskrit", slp1, "translate_sa"),
                                          ("which language is this", "English", "language")):
                    out_f.write(json.dumps({"id": f"{base}_en", "task": task, "lang": "en",
                                            "instruction": instr, "response": resp,
                                            "source_text": e.lower(), **w}, ensure_ascii=False) + "\n")
        out_f.flush()
        made += 1
        if made % 50 == 0:
            print(json.dumps({"pairs_done": made, "of": len(idx)}), flush=True)
    out_f.close()
    rows = [json.loads(l) for l in manifest.open(encoding="utf-8") if l.strip()]
    ds = {"n_items": len(rows), "n_pairs_sampled": int(len(idx)),
          "tasks": sorted({r['task'] for r in rows}),
          "provenance": "Itihāsa train split (rahular/itihasa; aligned Sa↔En) — TRAIN pairs only; "
                        "TTS: indic-parler (Devanagari for sa = native phonetics; English voice for en)",
          "note": "translation supervision is the aligned human translation — never machine output"}
    (OUT / "datasheet.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ds, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    raise SystemExit(main(a.n))
