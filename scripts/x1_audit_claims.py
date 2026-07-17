"""X1 — audit pranava's own claims through the Navya-Nyāya epistemic layer (reuses pramana).

Runs the auditor over the project's live claims and writes a report. The point: the epistemic
"Layer C" independently reaches the same verdicts the empirical gates did — including refusing the
retracted E2 holism claim as savyabhichāra.
"""
from __future__ import annotations

import json
from pathlib import Path

from pramana.domain.models.nyaya_example import PramanaType

from pranava.pramana_layer.auditor import Claim, Evidence, audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pramana_layer"

CLAIMS = [
    Claim(
        "VP 1.1 identifies śabda-tattva with brahman (śabdādvaita).",
        [Evidence(PramanaType.SHABDA, "mūla 'anādinidhanaṃ brahma śabdatattvaṃ' + KG anchor @1.1",
                  strength=1.0),
         Evidence(PramanaType.ANUMANA, "commentary section 'The Word-Brahman' concurs", strength=0.9)],
    ),
    Claim(
        "vākya-meaning is grasped as pratibhā (VP 2.143).",
        [Evidence(PramanaType.SHABDA, "mūla 2.143 'pratibhā... vākyārtha iti tāmāhuḥ' (locus classicus)",
                  strength=1.0)],
    ),
    Claim(
        "Speech encoders resolve meaning more holistically than text on late-resolving sentences.",
        [Evidence(PramanaType.PRATYAKSHA,
                  "E2 verb_final subset HI 0.41 vs 0.17 (48 items, 2 templates → degenerate CV)",
                  supports=True, strength=0.3),
         Evidence(PramanaType.PRATYAKSHA,
                  "E7 matched-vocab powered re-run: effect -0.07, CI includes 0", supports=False,
                  strength=0.9)],
    ),
    Claim(
        "There is no speech-vs-text holism difference (corrected conclusion).",
        [Evidence(PramanaType.PRATYAKSHA, "E7 P2 null (-0.07); P1 sanity holds", strength=0.9),
         Evidence(PramanaType.ANUMANA, "E6 scaling replication agrees", strength=0.8)],
    ),
    Claim(
        "Speech carries prosodic information that text structurally cannot (E5).",
        [Evidence(PramanaType.PRATYAKSHA, "E5 speech acc 1.0 vs text 0.5 (definitional gap 0.5)",
                  strength=0.85)],
    ),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = [audit(c).to_dict() for c in CLAIMS]
    (OUT / "audit.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    for r in reports:
        mark = {"ascertained": "✓", "not_ascertained": "✗", "doubtful": "?"}[r["verdict"]]
        fall = (" [" + ",".join(f["kind"] for f in r["hetvabhasa"]) + "]") if r["hetvabhasa"] else ""
        print(f"  {mark} {r['verdict']:16s}{fall}  {r['claim'][:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
