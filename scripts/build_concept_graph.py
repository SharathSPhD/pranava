"""Emit the verse-anchored concept graph to data/vakyapadiya/concept_graph.json."""
from __future__ import annotations

import json
from pathlib import Path

from pranava.kg.concepts import build_concept_graph

OUT = Path(__file__).resolve().parents[1] / "data" / "vakyapadiya" / "concept_graph.json"


def main() -> None:
    g = build_concept_graph()
    d = g.to_dict()
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"concepts={len(d['concepts'])} edges={len(d['edges'])} "
          f"curated={len(d['curated_relations'])} -> {OUT}")


if __name__ == "__main__":
    main()
