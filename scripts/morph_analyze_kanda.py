"""Run Saṃsādhanī morphological analysis over a kāṇḍa and report honest coverage.

Tokenisation is whitespace + daṇḍa/number stripping (no sandhi splitter — see
research/01-samsaadhanii-integration.md). Coverage = fraction of surface tokens
that receive >=1 valid morphological analysis. Nothing is invented.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pranava.corpus.edition import build_edition
from pranava.corpus.morph import SamsaadhaniiClient, service_available
from pranava.corpus.translit import strip_verse_markers

OUT = Path(__file__).resolve().parents[1] / "data" / "vakyapadiya"
_TOKEN_SPLIT = re.compile(r"\s+")


def tokens(line: str) -> list[str]:
    cleaned = strip_verse_markers(line)
    return [t for t in _TOKEN_SPLIT.split(cleaned) if t.strip()]


def main(kanda: int = 1) -> int:
    if not service_available():
        print("samsaadhanii service not reachable on :8090", file=sys.stderr)
        return 2
    ed = build_edition()
    client = SamsaadhaniiClient()
    verses = [k for k in ed.karikas if k.vid.kanda == kanda]

    rows = []
    tok_total = tok_covered = 0
    for k in verses:
        for line in k.mula_lines:
            for tok in tokens(line):
                analyses = client.analyze(tok)
                valid = [a for a in analyses if a.is_valid]
                tok_total += 1
                tok_covered += 1 if valid else 0
                rows.append(
                    {
                        "vid": k.canonical,
                        "token": tok,
                        "token_wx": client.to_wx(tok),
                        "n_analyses": len(valid),
                        "analyses": [
                            {"app": a.app, "root": a.root, "features": a.features} for a in valid
                        ],
                    }
                )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"morph_kanda{kanda}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    report = {
        "kanda": kanda,
        "verses": len(verses),
        "tokens": tok_total,
        "tokens_covered": tok_covered,
        "coverage": (tok_covered / tok_total) if tok_total else 0.0,
        "note": "whitespace tokenisation; no sandhi splitter (Heritage backend absent)",
    }
    (OUT / f"morph_kanda{kanda}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sys.exit(main(k))
