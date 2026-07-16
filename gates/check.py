"""Milestone gate checker — the machine that decides whether a milestone is really DONE.

Adopts prabodha's dual-verdict discipline: each gate yields a `code_gate` (does the
software work: tests, artifacts, schemas) and a `domain_gate` (does it meet the
*domain* standard: real coverage, scholarly integrity, honest reporting). A milestone
closes only when BOTH pass. Results are written to `gates/gate_<id>.json`.

Run: `python gates/check.py [gate_id ...]`  (no args → all). Nonzero exit if any fail.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES_DIR = ROOT / "gates"
PY = str(ROOT / ".venv" / "bin" / "python")


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def _verdict(passed: bool, evidence: str) -> dict:
    return {"verdict": "pass" if passed else "fail", "evidence": evidence}


# --------------------------------------------------------------------------- M0
def gate_M0() -> dict:
    rc, out = _run([PY, "-m", "pytest", "tests/corpus/test_verse_id.py",
                    "tests/corpus/test_edition.py", "-q"])
    code = _verdict(rc == 0, "verse_id + edition tests green" if rc == 0 else out)

    cov_p = ROOT / "data/vakyapadiya/coverage.json"
    ed_p = ROOT / "data/vakyapadiya/edition.jsonl"
    if cov_p.exists() and ed_p.exists():
        cov = json.loads(cov_p.read_text())
        n = sum(1 for _ in ed_p.open())
        checks = {
            "karikas==1797": cov.get("karikas") == 1797,
            "lines==1797": n == 1797,
            "translated_fraction==1.0": cov.get("translated_fraction") == 1.0,
            "contested==286": cov.get("contested") == 286,
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items()))
    else:
        dom = _verdict(False, "edition artifacts missing")
    return {"code_gate": code, "domain_gate": dom}


# --------------------------------------------------------------------------- M1
def gate_M1() -> dict:
    rc, out = _run([PY, "-m", "pytest", "tests/corpus/test_translit.py", "-q"])
    code = _verdict(rc == 0, "transliteration tests green" if rc == 0 else out)

    ed_p = ROOT / "data/vakyapadiya/edition.jsonl"
    if ed_p.exists():
        n = missing = 0
        for line in ed_p.open(encoding="utf-8"):
            row = json.loads(line)
            n += 1
            iast = row.get("iast_lines")
            if not iast or len(iast) != len(row["mula_lines"]) or not all(s.strip() for s in iast):
                missing += 1
        dom = _verdict(n == 1797 and missing == 0, f"rows={n}; missing_iast={missing}")
    else:
        dom = _verdict(False, "edition.jsonl missing")
    return {"code_gate": code, "domain_gate": dom}


# --------------------------------------------------------------------------- M2
def gate_M2() -> dict:
    """Morphological analysis pipeline (rescoped: morphology only; segmenter absent).

    code_gate: morph client tests green.
    domain_gate: real coverage on Brahma-kāṇḍa recorded and above an honest floor
                 (whitespace tokenisation; the floor reflects unsplit sandhi/compounds,
                 not fabricated success).
    """
    rc, out = _run([PY, "-m", "pytest", "tests/corpus/test_morph.py", "-q"])
    code = _verdict(rc == 0, "morph client tests green (or skipped if service down)"
                    if rc == 0 else out)

    rep_p = ROOT / "data/vakyapadiya/morph_kanda1_report.json"
    if rep_p.exists():
        rep = json.loads(rep_p.read_text())
        cov = rep.get("coverage", 0.0)
        # Honest floor: whitespace tokens that ARE full words should analyse.
        # We require the run to have actually executed over the whole kāṇḍa and
        # reached a documented, non-trivial coverage. The number is reported, not asserted true.
        ok = rep.get("verses", 0) >= 137 and rep.get("tokens", 0) > 300 and cov >= 0.30
        dom = _verdict(ok, f"verses={rep.get('verses')}, tokens={rep.get('tokens')}, "
                           f"coverage={cov:.3f} (whitespace tok; no segmenter)")
    else:
        dom = _verdict(False, "morph_kanda1_report.json missing — run scripts/morph_analyze_kanda.py")
    return {"code_gate": code, "domain_gate": dom}


GATES = {"M0": gate_M0, "M1": gate_M1, "M2": gate_M2}


def main(argv: list[str]) -> int:
    requested = argv or list(GATES)
    failed = 0
    for gid in requested:
        fn = GATES.get(gid)
        if fn is None:
            print(f"[ ?? ] {gid}: no such gate (defined: {', '.join(GATES)})")
            failed += 1
            continue
        result = fn()
        both_pass = all(g["verdict"] == "pass" for g in result.values())
        (GATES_DIR / f"gate_{gid}.json").write_text(
            json.dumps({"milestone": gid, **result,
                        "closed": both_pass}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        status = "PASS" if both_pass else "FAIL"
        cg, dg = result["code_gate"], result["domain_gate"]
        print(f"[{status}] {gid}  code:{cg['verdict']} domain:{dg['verdict']}")
        print(f"        code:   {cg['evidence'][:120]}")
        print(f"        domain: {dg['evidence'][:120]}")
        failed += not both_pass
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
