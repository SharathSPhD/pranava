"""Prepare the Vāgdhenu Sanskrit chant corpus (prathoshap/vagdhenu-data, CC-BY-4.0) — second public domain.

REAL human Sanskrit chant (pārāyaṇa; single scholar, two styles) with native SLP1 transcripts.
Split is defined transparently and leakage-safely BY SESSION (matching the Su-śrotā project's own
convention): sessions are sorted, and every 7th session (index % 7 == 0) is TEST — deterministic,
publishable, no cherry-picking. ~85/15 train/test by construction.

Writes data/alm/vagdhenu/{wav→symlinks? no: copies paths in manifest referencing the raw dir}
Actually: manifest points at the RAW wav files (no copy — they're already on disk), plus datasheet.

Run: .venv/bin/python scripts/alm/prep_vagdhenu.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/vagdhenu_raw"
OUT = ROOT / "data/alm/vagdhenu"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out = []
    sessions = set()
    for style_dir in sorted(RAW.glob("style_*")):
        meta = style_dir / "metadata.csv"
        if not meta.exists():
            continue
        with meta.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                wav = style_dir / r["file_name"]
                if not wav.exists():
                    continue
                sess = f"{style_dir.name}:{r.get('session', '?')}"
                sessions.add(sess)
                rows_out.append({"style": style_dir.name, "session": sess,
                                 "text": r["text_slp1"].strip(),
                                 "text_devanagari": r["text_devanagari"].strip(),
                                 "meter": r.get("meter", ""),
                                 "duration_s": float(r.get("duration", 0) or 0),
                                 "wav": str(wav.relative_to(ROOT))})
    sess_sorted = sorted(sessions)
    test_sessions = {s for i, s in enumerate(sess_sorted) if i % 7 == 0}
    n = {"train": 0, "test": 0}
    for i, r in enumerate(rows_out):
        split = "test" if r["session"] in test_sessions else "train"
        r["split"] = split
        r["id"] = f"vg_{split[:2]}_{n[split]:05d}"
        n[split] += 1
    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows_out) + "\n", encoding="utf-8")
    hours = {s: round(sum(r["duration_s"] for r in rows_out if r["split"] == s) / 3600, 2)
             for s in ("train", "test")}
    ds = {"counts": n, "hours": hours, "n_sessions": len(sess_sorted),
          "n_test_sessions": len(test_sessions),
          "split_rule": "sessions sorted; every 7th session (idx%7==0) = TEST — deterministic, "
                        "leakage-safe (Su-śrotā convention: hold out by session), no cherry-picking",
          "provenance": "prathoshap/vagdhenu-data (CC-BY-4.0) — REAL human Sanskrit chant, native SLP1",
          "note": "single-speaker corpus: measures chant-domain accuracy, NOT speaker generalization"}
    (OUT / "datasheet.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ds, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
