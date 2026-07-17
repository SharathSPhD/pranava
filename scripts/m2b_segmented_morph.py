"""M2b — sandhi-split then morphologically analyse the Brahma-kāṇḍa; report honest coverage.

Pipeline: vidyut-cheda segments each mūla line into pādas → each pāda is validated by the
authoritative Saṃsādhanī morph.cgi. Coverage = fraction of segmented pādas that receive a valid
morphological analysis. vidyut's segmentation errors fail validation, so the number stays honest.
Compares against the M2 whitespace baseline (0.455).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pranava.corpus.edition import build_edition
from pranava.corpus.morph import SamsaadhaniiClient, service_available
from pranava.corpus.segmenter import data_available, segment_line

OUT = Path(__file__).resolve().parents[1] / "data" / "vakyapadiya"


def main(kanda: int = 1) -> int:
    if not data_available():
        print("vidyut data missing", file=sys.stderr)
        return 2
    if not service_available():
        print("samsaadhanii not reachable", file=sys.stderr)
        return 2

    ed = build_edition()
    client = SamsaadhaniiClient()
    verses = [k for k in ed.karikas if k.vid.kanda == kanda]

    rows = []
    pada_total = pada_valid = 0
    for k in verses:
        for line in k.mula_lines:
            for p in segment_line(line):
                analyses = [a for a in client.analyze(p.text_dev) if a.is_valid]
                pada_total += 1
                pada_valid += 1 if analyses else 0
                rows.append({
                    "vid": k.canonical, "pada": p.text_dev, "vidyut_lemma": p.lemma,
                    "n_analyses": len(analyses),
                    "roots": [a.root for a in analyses][:3],
                })

    coverage = (pada_valid / pada_total) if pada_total else 0.0
    (OUT / f"m2b_segmented_kanda{kanda}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    report = {
        "kanda": kanda, "verses": len(verses),
        "segmented_padas": pada_total, "padas_validated": pada_valid,
        "coverage_segmented": round(coverage, 4),
        "m2_whitespace_baseline": 0.455,
        "lift": round(coverage - 0.455, 4),
        "note": "vidyut-cheda split + Saṃsādhanī morph validation; errors fail validation (honest).",
    }
    (OUT / f"m2b_report_kanda{kanda}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
