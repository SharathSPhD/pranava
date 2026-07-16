"""Milestone gate checker — the machine that says whether a milestone is really DONE.

Each gate is a callable returning (passed: bool, evidence: str). No milestone is
DONE unless its gate passes here. Run: `python gates/check.py [gate_id ...]`.
Exit code is nonzero if any requested gate fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def gate_M0() -> tuple[bool, str]:
    """Aligned edition builds & is fully tested."""
    py = str(ROOT / ".venv" / "bin" / "python")
    rc, out = _run([py, "-m", "pytest", "tests/corpus", "-q"])
    if rc != 0:
        return False, f"pytest tests/corpus failed:\n{out}"
    cov_path = ROOT / "data" / "vakyapadiya" / "coverage.json"
    ed_path = ROOT / "data" / "vakyapadiya" / "edition.jsonl"
    if not cov_path.exists() or not ed_path.exists():
        return False, "edition artifacts missing — run scripts/build_edition.py"
    cov = json.loads(cov_path.read_text())
    n_lines = sum(1 for _ in ed_path.open())
    checks = {
        "karikas==1797": cov.get("karikas") == 1797,
        "edition.jsonl lines==1797": n_lines == 1797,
        "translated_fraction==1.0": cov.get("translated_fraction") == 1.0,
        "contested==286": cov.get("contested") == 286,
    }
    ok = all(checks.values())
    return ok, "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())


def gate_M1() -> tuple[bool, str]:
    """IAST transliteration + Devanāgarī integrity."""
    py = str(ROOT / ".venv" / "bin" / "python")
    rc, out = _run([py, "-m", "pytest", "tests/corpus/test_translit.py", "-q"])
    if rc != 0:
        return False, f"transliteration tests failed:\n{out}"
    ed_path = ROOT / "data" / "vakyapadiya" / "edition.jsonl"
    if not ed_path.exists():
        return False, "edition.jsonl missing"
    n = missing_iast = 0
    for line in ed_path.open(encoding="utf-8"):
        row = json.loads(line)
        n += 1
        iast = row.get("iast_lines")
        if not iast or len(iast) != len(row["mula_lines"]) or not all(s.strip() for s in iast):
            missing_iast += 1
    ok = n == 1797 and missing_iast == 0
    return ok, f"rows={n}; rows_missing_iast={missing_iast}"


GATES = {"M0": gate_M0, "M1": gate_M1}


def main(argv: list[str]) -> int:
    requested = argv or list(GATES)
    failed = 0
    for gid in requested:
        fn = GATES.get(gid)
        if fn is None:
            print(f"[ ?? ] {gid}: no such gate (defined: {', '.join(GATES)})")
            failed += 1
            continue
        passed, evidence = fn()
        print(f"[{'PASS' if passed else 'FAIL'}] {gid}: {evidence}")
        failed += not passed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
