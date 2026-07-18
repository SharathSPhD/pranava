"""Precompute Parakeet encoder features for every XL-corpus clip → speech_corpus_indic_xl/feats/*.npy.

Training iterates the corpus many times; encoding once (0.6B Parakeet) and caching (T,1024) float16
arrays makes each epoch cheap. Resumable — skips clips whose .npy already exists. Runs in the NeMo
container on either machine:  scripts/alm/in_container.sh python /work/pranava/scripts/alm/precompute_feats_xl.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
XL = ROOT / "data/alm/speech_corpus_indic_xl"
FEATS = XL / "feats"


def main() -> int:
    from pranava.alm.data import read_wav
    from pranava.alm.encoder import ParakeetEncoder

    FEATS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (XL / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    enc = ParakeetEncoder().load()
    done = skipped = 0
    for r in rows:
        out = FEATS / f"{r['id']}.npy"
        if out.exists():
            skipped += 1
            continue
        wav, sr = read_wav(ROOT / r["wav"])
        f = enc.encode(wav, sr=sr)
        f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
        np.save(out, np.squeeze(f).astype(np.float16))
        done += 1
        if done % 200 == 0:
            print(json.dumps({"encoded": done, "skipped": skipped, "of": len(rows)}), flush=True)
    print(json.dumps({"encoded": done, "skipped": skipped, "total": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
