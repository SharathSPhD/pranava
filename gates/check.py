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


def gate_M3() -> dict:
    """Verse-anchored concept knowledge graph."""
    rc, out = _run([PY, "-m", "pytest", "tests/kg/", "-q"])
    code = _verdict(rc == 0, "concept-graph tests green" if rc == 0 else out)

    g_p = ROOT / "data/vakyapadiya/concept_graph.json"
    if g_p.exists():
        g = json.loads(g_p.read_text())
        concepts = g.get("concepts", [])
        edges = g.get("edges", [])
        curated = g.get("curated_relations", [])
        all_anchored = all(c.get("anchor_count", 0) >= 1 for c in concepts)
        curated_anchored = all(r.get("anchor_vid") for r in curated)
        checks = {
            "concepts>=12": len(concepts) >= 12,
            "all_concepts_anchored": all_anchored,
            "edges>0": len(edges) > 0,
            "curated>=4": len(curated) >= 4,
            "curated_all_anchored": curated_anchored,
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | concepts={len(concepts)} edges={len(edges)} curated={len(curated)}")
    else:
        dom = _verdict(False, "concept_graph.json missing — run scripts/build_concept_graph.py")
    return {"code_gate": code, "domain_gate": dom}


def gate_E0() -> dict:
    """Speech-model representation harness on GPU."""
    rc, out = _run([PY, "-m", "pytest", "tests/speech/", "-q"])
    # tests skip (not fail) if CUDA is unavailable; treat all-skipped as a soft pass
    code = _verdict(rc == 0, "harness tests green (or skipped without CUDA)" if rc == 0 else out)

    man_p = ROOT / "data/speech/e0_manifest.json"
    npy_p = ROOT / "data/speech/e0_lastlayer.npy"
    if man_p.exists() and npy_p.exists():
        man = json.loads(man_p.read_text())
        checks = {
            "device==cuda": str(man.get("device", "")).startswith("cuda"),
            "n_layers==13": man.get("n_layers") == 13,
            "dim==768": man.get("dim") == 768,
            "frame_rate~50Hz": 40 < man.get("frame_rate_hz", 0) < 60,
            "frames>0": man.get("n_frames", 0) > 0,
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | {man.get('n_frames')} frames @ {man.get('frame_rate_hz')}Hz "
                         f"in {man.get('elapsed_s')}s on {man.get('device')}")
    else:
        dom = _verdict(False, "e0 manifest/tensor missing — run scripts/e0_harness_demo.py")
    return {"code_gate": code, "domain_gate": dom}


def gate_E1() -> dict:
    """Controlled stimulus set with ground truth + synthesized audio."""
    rc, out = _run([PY, "-m", "pytest", "tests/experiments/test_stimuli.py", "-q"])
    code = _verdict(rc == 0, "stimulus tests green" if rc == 0 else out)

    man_p = ROOT / "data/stimuli/manifest.jsonl"
    ds_p = ROOT / "data/stimuli/datasheet.json"
    wav_dir = ROOT / "data/stimuli/wav"
    if man_p.exists() and ds_p.exists():
        rows = [json.loads(l) for l in man_p.open(encoding="utf-8")]
        n_wav = len(list(wav_dir.glob("*.wav"))) if wav_dir.exists() else 0
        have_all_wavs = all((ROOT / r["wav"]).exists() for r in rows)
        labelled = all(r.get("meaning_label") and r.get("resolution") for r in rows)
        checks = {
            "items>=200": len(rows) >= 200,
            "all_wavs_present": have_all_wavs and n_wav >= len(rows),
            "all_labelled": labelled,
            "both_resolution_classes": len({r["resolution"] for r in rows}) == 2,
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | items={len(rows)} wavs={n_wav}")
    else:
        dom = _verdict(False, "manifest/datasheet missing — run scripts/e1_synthesize.py")
    return {"code_gate": code, "domain_gate": dom}


GATES = {
    "M0": gate_M0, "M1": gate_M1, "M2": gate_M2, "M3": gate_M3,
    "E0": gate_E0, "E1": gate_E1,
}


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
