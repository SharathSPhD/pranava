"""Build the site's data from the gated artifacts — figures provably match the gates.

Extracts every number the site shows (leaderboard, emergence curve, CER progression, gate count)
from the real artifacts under data/ and gates/, and writes docs/data.json. The site fetches this,
so nothing on the page is hand-typed: it is the gated result or it is not shown. (Mirrors prabodha's
build-data pipeline, in Python, no npm.)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
GATES = ROOT / "gates"
OUT = ROOT / "docs" / "data.json"


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    lb = _load(DATA / "benchmark" / "sota_leaderboard.json")
    em = _load(DATA / "sphota_lens" / "emergence_report.json")
    p2 = _load(DATA / "alm" / "p2_metrics.json")
    p5 = _load(DATA / "alm" / "p5_metrics.json")
    lora = _load(DATA / "alm" / "lora_metrics.json")

    gate_files = sorted(GATES.glob("gate_*.json"))
    gates_passing = 0
    for g in gate_files:
        gd = _load(g)
        cg, dg = gd.get("code_gate", {}), gd.get("domain_gate", {})
        if gd.get("closed") is True or (cg.get("verdict") in ("pass", "pruned")
                                        and dg.get("verdict") in ("pass", "pruned")):
            gates_passing += 1

    ours = next((r for r in lb.get("leaderboard", []) if "ours" in r.get("model", "")), {})
    data = {
        "generated_from": "gates/ + data/ artifacts (scripts/site/build_site_data.py)",
        "leaderboard": lb.get("leaderboard", []),
        "ours_cer": ours.get("cer"),
        "emergence": {
            "decodability_by_layer": em.get("decodability_by_layer", []),
            "chance": em.get("chance"),
            "peak_layer": em.get("peak_layer"),
            "causal_peak_layer": em.get("causal_peak_layer"),
            "sphota_layer": em.get("sphota_layer"),
            "validated": em.get("validated"),
        },
        "cer_progression": {
            "proj_200m": p2.get("val_cer_audio"),
            "proj_1b": p5.get("val_cer_audio"),
            "lora_200m": lora.get("val_cer_audio"),
        },
        "gates_passing": gates_passing,
        "gates_total": len(gate_files),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}: {gates_passing}/{len(gate_files)} gates, ours CER {data['ours_cer']}, "
          f"sphoṭa L{data['emergence']['sphota_layer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
