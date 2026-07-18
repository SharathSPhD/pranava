"""Corpus v2: expand the native-Sanskrit speech corpus from PSALM's full paninian_v1 fixture.

The v1 corpus used only the first 600 unique sentences of PSALM's 10,000-row gold-kāraka fixture —
data, not architecture, is the specialist's biggest honest deficit (0.335 h of speech). This script
TTSes the NEXT unique sentences (skipping every text already in v1) with the same indic-parler-tts
voice and writes them into data/alm/speech_corpus_indic_xl/ as ADDITIONAL TRAIN items. The v1 58-clip
val split is FROZEN and carried over verbatim — the benchmark stays comparable.

Supports range sharding so GB10 and the 5090 can synthesize in parallel:
  python scripts/alm/expand_corpus_indic.py --start 0 --n 2400            # GB10
  python scripts/alm/expand_corpus_indic.py --start 2400 --n 2400 ...     # 5090 (own checkout)
Then merge shards with --merge.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data/alm/speech_corpus_indic"
XL = ROOT / "data/alm/speech_corpus_indic_xl"
FIXTURE = next((p for p in (
    Path("/home/sharaths/projects/PSALM/data/fixtures/paninian_v1.jsonl"),
    Path("/work/PSALM/data/fixtures/paninian_v1.jsonl"),
    Path("/fusion-project/PSALM/data/fixtures/paninian_v1.jsonl"),  # rtx5090-train container
) if p.exists()), Path("/home/sharaths/projects/PSALM/data/fixtures/paninian_v1.jsonl"))
DESC = "A clear male voice speaks slowly and distinctly with a neutral tone in a quiet room."


def fresh_sentences() -> list[dict]:
    """All TTS-friendly unique fixture sentences NOT already in the v1 corpus, in fixture order."""
    used = {json.loads(x)["text"] for x in (V1 / "manifest.jsonl").open(encoding="utf-8") if x.strip()}
    seen, rows = set(used), []
    for line in FIXTURE.open(encoding="utf-8"):
        r = json.loads(line)
        text = r.get("text", "").strip()
        if not text or text in seen or not (8 <= len(text) <= 80):
            continue
        seen.add(text)
        rows.append({"text": text, "karaka": r.get("karaka_parse", []), "source": "paninian_v1"})
    return rows


def synth(start: int, n: int) -> int:
    import numpy as np
    import soundfile as sf
    import torch
    from indic_transliteration import sanscript

    snaps = Path(os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--indic-parler-tts/snapshots"))
    local = next(snaps.glob("*/"), None) if snaps.exists() else None
    if local is not None:  # GB10: cached snapshot, stay offline
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    src = str(local) if local is not None else "ai4bharat/indic-parler-tts"  # 5090: download once
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(src).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(src)
    sr = int(model.config.sampling_rate)
    wav_dir = XL / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    desc_ids = tok(DESC, return_tensors="pt").to("cuda")

    batch = fresh_sentences()[start:start + n]
    shard = XL / f"shard_{start:06d}.jsonl"
    done_ids = set()
    if shard.exists():  # resumable
        done_ids = {json.loads(x)["id"] for x in shard.open(encoding="utf-8") if x.strip()}
    out = shard.open("a", encoding="utf-8")
    import time
    t0 = time.time()
    for i, r in enumerate(batch):
        cid = f"x{start + i:05d}"
        if cid in done_ids:
            continue
        dev_text = sanscript.transliterate(r["text"], sanscript.SLP1, sanscript.DEVANAGARI)
        p = tok(dev_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            aud = model.generate(input_ids=desc_ids.input_ids, attention_mask=desc_ids.attention_mask,
                                 prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        a = np.atleast_1d(aud.cpu().numpy().squeeze().astype(np.float32))
        if a.ndim != 1 or len(a) < int(0.3 * sr):  # degenerate synthesis — skip, honestly logged
            print(json.dumps({"skip": cid, "reason": f"degenerate audio shape={a.shape}",
                              "text": r["text"]}), flush=True)
            continue
        wp = wav_dir / f"{cid}.wav"
        sf.write(str(wp), a, sr)
        row = {**r, "id": cid, "text_devanagari": dev_text, "wav": str(wp.relative_to(ROOT)),
               "sr": sr, "duration_s": round(len(a) / sr, 3), "split": "train"}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(json.dumps({"done": i + 1, "of": len(batch), "clips_per_s": round(rate, 2),
                              "eta_min": round((len(batch) - i - 1) / rate / 60, 1)}), flush=True)
    out.close()
    print(f"shard {shard.name} complete")
    return 0


def merge() -> int:
    """v1 manifest (600, val frozen) + all XL shards → XL manifest + datasheet."""
    rows = [json.loads(x) for x in (V1 / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    n_v1 = len(rows)
    for shard in sorted(XL.glob("shard_*.jsonl")):
        rows += [json.loads(x) for x in shard.open(encoding="utf-8") if x.strip()]
    (XL / "manifest.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    hours = round(sum(r.get("duration_s", 0) for r in rows) / 3600, 3)
    n_val = sum(1 for r in rows if r["split"] == "val")
    ds = {"n_items": len(rows), "n_train": len(rows) - n_val, "n_val": n_val, "total_hours": hours,
          "provenance": f"v1 corpus ({n_v1} items, 58-clip val FROZEN) + expansion shards from the "
                        "remaining unique sentences of PSALM paninian_v1.jsonl (gold kāraka by "
                        "construction), same indic-parler-tts voice/protocol",
          "note": "expansion adds TRAIN only; val is byte-identical to v1 so benchmarks stay comparable"}
    (XL / "datasheet.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ds, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    raise SystemExit(merge() if a.merge else synth(a.start, a.n))
