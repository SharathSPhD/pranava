"""Build the aligned Vākyapadīya edition to data/vakyapadiya/edition.jsonl + report."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pranava.corpus.edition import build_edition
from pranava.corpus.translit import to_iast

OUT = Path(__file__).resolve().parents[1] / "data" / "vakyapadiya"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ed = build_edition()
    cov = ed.coverage()

    with (OUT / "edition.jsonl").open("w", encoding="utf-8") as f:
        for k in ed.karikas:
            row = asdict(k)
            row["vid"] = k.vid.canonical
            row["kanda"] = k.vid.kanda
            row["mula_lines"] = list(k.mula_lines)
            row["iast_lines"] = [to_iast(line) for line in k.mula_lines]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (OUT / "coverage.json").write_text(json.dumps(cov, indent=2), encoding="utf-8")
    print(json.dumps(cov, indent=2))


if __name__ == "__main__":
    main()
