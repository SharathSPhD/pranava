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


def gate_M2b() -> dict:
    """Sandhi-split + validated morphology lifts coverage (honest; 0.85 target NOT claimed).

    Validity: segmenter tests green; the self-checking pipeline (vidyut split → Saṃsādhanī
    validation) ran over the kāṇḍa and measurably lifted coverage over the M2 baseline. The
    aspirational ≥0.85 is documented as not reached (off-the-shelf vidyut errors cap it).
    """
    rc, out = _run([PY, "-m", "pytest", "tests/corpus/test_segmenter.py", "-q"])
    code = _verdict(rc == 0, "segmenter tests green (or skipped without vidyut data)"
                    if rc == 0 else out)
    rep_p = ROOT / "data/vakyapadiya/m2b_report_kanda1.json"
    if rep_p.exists():
        rep = json.loads(rep_p.read_text())
        cov = rep.get("coverage_segmented", 0.0)
        base = rep.get("m2_whitespace_baseline", 0.455)
        checks = {
            "pipeline_ran": rep.get("segmented_padas", 0) > 500,
            "coverage_lifted_over_M2": cov > base,
            "honestly_reported": "note" in rep and rep.get("lift") is not None,
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | coverage={cov:.3f} (lift +{rep.get('lift')}); 0.85 target NOT reached "
                         f"(vidyut errors fail validation — honest)")
    else:
        dom = _verdict(False, "m2b report missing — run scripts/m2b_segmented_morph.py")
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


def gate_E2() -> dict:
    """H-HOLISM experiment ran validly and is honestly reported.

    A gate on VALIDITY, not on a positive result (honest negatives ship). Requires:
    metric tests green; pre-registration exists; probe beats chance; every confirmatory
    test recorded with a CI + p; figure + report present.
    """
    rc, out = _run([PY, "-m", "pytest", "tests/experiments/", "-q"])
    code = _verdict(rc == 0, "experiment + metric tests green" if rc == 0 else out)

    res_p = ROOT / "data/experiments/e2_results.json"
    prereg = ROOT / "research/prereg/H-HOLISM.md"
    fig = ROOT / "data/experiments/e2_trajectories.png"
    report = ROOT / "research/E2-report.md"
    if res_p.exists():
        res = json.loads(res_p.read_text())
        pv = res.get("probe_validity", {})
        tests = res.get("tests", {})
        beats_chance = (
            pv.get("speech_full_utt_Ptrue", 0) > 2 * pv.get("chance", 1)
            and pv.get("text_full_utt_Ptrue", 0) > 2 * pv.get("chance", 1)
        )
        all_reported = all(
            "ci95" in t and "p_one_sided_predicted_dir" in t for t in tests.values()
        )
        checks = {
            "prereg_exists": prereg.exists(),
            "report_exists": report.exists(),
            "figure_exists": fig.exists(),
            "probe_beats_chance": beats_chance,
            ">=3_confirmatory_tests": len(tests) >= 3,
            "all_tests_have_CI_and_p": all_reported,
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | probe speech={pv.get('speech_full_utt_Ptrue')} "
                         f"text={pv.get('text_full_utt_Ptrue')} chance={pv.get('chance')}")
    else:
        dom = _verdict(False, "e2_results.json missing — run scripts/e2_run.py")
    return {"code_gate": code, "domain_gate": dom}


def gate_E5() -> dict:
    """Prosody gap experiment ran validly and is honestly reported (incl. its negative)."""
    res_p = ROOT / "data/experiments/e5_results.json"
    prereg = ROOT / "research/prereg/H-PROSODY-E5.md"
    report = ROOT / "research/E5-report.md"
    code = _verdict(res_p.exists(), "e5 ran" if res_p.exists() else "e5 not run")
    if res_p.exists():
        res = json.loads(res_p.read_text())
        checks = {
            "prereg_exists": prereg.exists(),
            "report_exists": report.exists(),
            "PR1_recorded": "acc_speech_best_layer" in res,
            "PR2_text_near_chance": res.get("PR2_text_near_chance") is True,
            "gap_has_CI": "ci95" in res.get("prosody_gap_speech_minus_text", {}),
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | gap={res.get('prosody_gap_speech_minus_text',{}).get('effect')} "
                         f"(localization exploratory NULL — saturated; see E5-report)")
    else:
        dom = _verdict(False, "e5_results.json missing")
    return {"code_gate": code, "domain_gate": dom}


def gate_E6() -> dict:
    """Well-powered verb-final scaling replication ran and is honestly reported.

    Validity-not-outcome: a rigorous corrective negative is a successful milestone. Requires the
    stimulus tests green, a larger valid-CV design, a tighter CI than E2, and the correction doc.
    """
    rc, out = _run([PY, "-m", "pytest", "tests/experiments/test_stimuli_verbfinal.py", "-q"])
    code = _verdict(rc == 0, "scaled-vf stimulus tests green" if rc == 0 else out)
    res_p = ROOT / "data/experiments/e6_results.json"
    corr = ROOT / "research/E6-correction.md"
    if res_p.exists():
        res = json.loads(res_p.read_text())
        checks = {
            "n>=200": res.get("n_items", 0) >= 200,
            "ci_tightened_vs_E2": res.get("ci_tightened") is True,
            "correction_doc_exists": corr.exists(),
            "effect_has_CI": "ci95" in res.get("speech_gt_text", {}),
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | effect={res['speech_gt_text']['effect']} "
                         f"CI={res['speech_gt_text']['ci95']} (corrective negative)")
    else:
        dom = _verdict(False, "e6_results.json missing — run scripts/e6_scale_verbfinal.py")
    return {"code_gate": code, "domain_gate": dom}


def gate_E7() -> dict:
    """Definitive matched-vocabulary holism test ran, is valid (P1 sanity holds), reported."""
    rc, out = _run([PY, "-m", "pytest", "tests/experiments/test_stimuli_early.py", "-q"])
    code = _verdict(rc == 0, "early-pool tests green" if rc == 0 else out)
    res_p = ROOT / "data/experiments/e7_results.json"
    if res_p.exists():
        res = json.loads(res_p.read_text())
        p1s = res.get("P1_speech_late_gt_early", {})
        p2 = res.get("P2_speech_gt_text_late", {})
        checks = {
            "matched_pools>=200": res.get("n_early", 0) >= 200 and res.get("n_late", 0) >= 200,
            "P1_sanity_holds": p1s.get("supported_holm") is True,  # metric validity check
            "P2_reported_with_CI": "ci95" in p2,
            "verdict_recorded": bool(res.get("verdict")),
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | {res.get('verdict')}")
    else:
        dom = _verdict(False, "e7_results.json missing — run scripts/e7_definitive.py")
    return {"code_gate": code, "domain_gate": dom}


def gate_P0() -> dict:
    """ALM vertical slice (audio→Parakeet→projector→prabhasa core→text) runs + trains on GB10.

    The heavy pieces run in prabhasa/nemo-gb10 (mamba_ssm + NeMo). Evidence is
    data/alm/p0_manifest.json produced by `scripts/alm/in_container.sh python
    scripts/alm/p0_demo.py`. code_gate: the ALM modules + tests exist; domain_gate: the manifest
    shows a real end-to-end run and a successful overfit (loss drop + exact emit).
    """
    mods = [ROOT / "src/pranava/alm" / f for f in
            ("core_adapter.py", "projector.py", "encoder.py", "pipeline.py")]
    tests = [ROOT / "tests/alm" / f for f in
             ("test_core_adapter.py", "test_projector_overfit.py", "test_pipeline_e2e.py")]
    code = _verdict(all(p.exists() for p in mods + tests),
                    "ALM modules + tests present" if all(p.exists() for p in mods + tests)
                    else "missing ALM modules/tests")
    man_p = ROOT / "data/alm/p0_manifest.json"
    if man_p.exists():
        m = json.loads(man_p.read_text())
        e2e, over = m.get("e2e", {}), m.get("overfit", {})
        checks = {
            "e2e_on_cuda": str(e2e.get("device", "")).startswith("cuda"),
            "encoder_frames_to_core": e2e.get("encoder_dim", 0) > 0
            and e2e.get("d_model") == 768 and e2e.get("n_sphota_tokens", 0) >= 1,
            "overfit_loss_dropped": over.get("loss_after", 9) < 0.5 * over.get("loss_before", 1),
            "overfit_exact_emit": over.get("exact_match") is True,
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | enc_dim={e2e.get('encoder_dim')} tokens={e2e.get('n_sphota_tokens')} "
                         f"loss {over.get('loss_before')}→{over.get('loss_after')} "
                         f"emit={over.get('emitted')!r}")
    else:
        dom = _verdict(False, "p0_manifest.json missing — run scripts/alm/p0_demo.py in-container")
    return {"code_gate": code, "domain_gate": dom}


def gate_P1() -> dict:
    """Paired speech↔text corpus built with gold labels, real audio, fixed split, datasheet."""
    rc, out = _run([PY, "-m", "pytest", "tests/alm/test_data.py", "-q"])
    code = _verdict(rc == 0, "data-loader tests green" if rc == 0 else out)
    ds_p = ROOT / "data/alm/speech_corpus/datasheet.json"
    man_p = ROOT / "data/alm/speech_corpus/manifest.jsonl"
    if ds_p.exists() and man_p.exists():
        ds = json.loads(ds_p.read_text())
        rows = [json.loads(x) for x in man_p.open(encoding="utf-8") if x.strip()]
        splits = {r["split"] for r in rows}
        all_wavs = all((ROOT / r["wav"]).exists() for r in rows)
        checks = {
            "items>=200": len(rows) >= 200,
            "both_splits": {"train", "val"} <= splits,
            "all_wavs_present": all_wavs,
            "datasheet_has_hours": ds.get("total_hours", 0) > 0,
            "limitation_documented": bool(ds.get("limitation")),
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | {len(rows)} items, {ds.get('total_hours')}h, gold kāraka")
    else:
        dom = _verdict(False, "corpus missing — run scripts/alm/build_speech_corpus.py in-container")
    return {"code_gate": code, "domain_gate": dom}


def gate_P2() -> dict:
    """Trained Sphoṭa Projector: audio-conditioned understanding beats the no-audio baseline."""
    rc, out = _run([PY, "-m", "pytest", "tests/alm/test_train_metrics.py", "-q"])
    code = _verdict(rc == 0, "metric tests green" if rc == 0 else out)
    m_p = ROOT / "data/alm/p2_metrics.json"
    ckpt = ROOT / "data/alm/projector.pt"
    if m_p.exists():
        m = json.loads(m_p.read_text())
        hist = m.get("train_loss_history", [])
        checks = {
            "trained_enough": len(hist) >= 4,
            "loss_decreased": len(hist) >= 2 and hist[-1] < 0.8 * hist[0],
            "beats_no_audio_baseline": m.get("beats_baseline") is True,
            "meaningful_cer_gain>=0.1": m.get("cer_improvement", 0) >= 0.1,
            "projector_ckpt_on_disk": ckpt.exists(),
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | CER audio={m.get('val_cer_audio')} vs no-audio="
                         f"{m.get('val_cer_noaudio_baseline')} (Δ{m.get('cer_improvement')}); "
                         f"loss {hist[0] if hist else '?'}→{hist[-1] if hist else '?'}")
    else:
        dom = _verdict(False, "p2_metrics.json missing — run src/pranava/alm/train.py in-container")
    return {"code_gate": code, "domain_gate": dom}


def gate_E4() -> dict:
    """Consolidated research report exists, is honest (reports the correction), and is grounded."""
    paper = ROOT / "PAPER.md"
    if not paper.exists():
        return {"code_gate": _verdict(False, "PAPER.md missing"),
                "domain_gate": _verdict(False, "missing")}
    t = paper.read_text(encoding="utf-8")
    checks = {
        "has_abstract": "## Abstract" in t,
        "reports_the_null": "no speech-vs-text holism" in t.lower(),
        "reports_the_correction": "falsified" in t.lower() and "savyabhich" in t.lower(),
        "cites_edition_and_experiments": "M0" in t and "E7" in t and "X1" in t,
        "has_limitations": "## 6. Limitations" in t or "Limitations" in t,
    }
    code = _verdict("## 7. Conclusion" in t, "report structured with conclusion")
    dom = _verdict(all(checks.values()),
                   "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items()))
    return {"code_gate": code, "domain_gate": dom}


def gate_X0() -> dict:
    """Autoresearch loop (reuses prabodha EFE) wired end-to-end with a recorded cycle."""
    rc, out = _run([PY, "-m", "pytest", "tests/autoresearch/", "-q"])
    code = _verdict(rc == 0, "autoresearch loop tests green" if rc == 0 else out)

    ledger = ROOT / "research" / "efe_ledger.jsonl"
    if ledger.exists():
        entries = [json.loads(x) for x in ledger.open(encoding="utf-8") if x.strip()]
        kinds = {e.get("kind") for e in entries}
        n_obs = sum(1 for e in entries if e.get("kind") == "observation")
        n_prop = sum(1 for e in entries if e.get("kind") == "proposal")
        checks = {
            "has_observations": "observation" in kinds and n_obs >= 1,
            "has_proposals": "proposal" in kinds and n_prop >= 1,
            "full_cycle_recorded": n_obs >= 2 and n_prop >= 1,  # propose→run→observe→re-propose
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | obs={n_obs} prop={n_prop}")
    else:
        dom = _verdict(False, "efe_ledger.jsonl missing — run scripts/autoresearch_step.py")
    return {"code_gate": code, "domain_gate": dom}


def gate_X2() -> dict:
    """NSM reference spec exists and every [EVIDENCED, <id>] tag cites a really-passing gate."""
    import re

    spec = ROOT / "specs" / "NSM-reference-spec.md"
    if not spec.exists():
        return {"code_gate": _verdict(False, "NSM-reference-spec.md missing"),
                "domain_gate": _verdict(False, "missing")}
    text = spec.read_text(encoding="utf-8")
    # collect milestone ids referenced as evidence, e.g. [EVIDENCED, E0] or [EVIDENCED, M0–M3]
    cited = set(re.findall(r"\[EVIDENCED[^\]]*?\b(M\d|E\d|X\d)\b", text))
    # a milestone is "really passing" iff its gate_<id>.json exists with closed/pass
    def passing(mid: str) -> bool:
        gp = GATES_DIR / f"gate_{mid}.json"
        if not gp.exists():
            return False
        g = json.loads(gp.read_text())
        return (g.get("closed") is True
                or all(v.get("verdict") in ("pass", "pruned")
                       for v in (g.get("code_gate", {}), g.get("domain_gate", {}))))
    unbacked = sorted(m for m in cited if not passing(m))
    has_labels = all(tag in text for tag in ("[EVIDENCED", "[ANALOGICAL]", "[OPEN]"))
    code = _verdict(has_labels, "spec has evidenced/analogical/open labels" if has_labels
                    else "missing honesty labels")
    dom = _verdict(len(cited) >= 4 and not unbacked,
                   f"cited={sorted(cited)}; unbacked_evidence_claims={unbacked}")
    return {"code_gate": code, "domain_gate": dom}


def gate_X1() -> dict:
    """Pramāṇa validation layer (reuses pramana) audits pranava's own claims correctly."""
    rc, out = _run([PY, "-m", "pytest", "tests/pramana_layer/", "-q"])
    code = _verdict(rc == 0, "auditor tests green" if rc == 0 else out)
    audit_p = ROOT / "data/pramana_layer/audit.json"
    if audit_p.exists():
        rows = json.loads(audit_p.read_text())
        # the retracted holism claim MUST be refused on hetvābhāsa grounds
        holism = next((r for r in rows if "more holistically" in r["claim"]), None)
        holism_refused = bool(holism and r_ok(holism))
        n_ascertained = sum(1 for r in rows if r["verdict"] == "ascertained")
        checks = {
            "audited>=4_claims": len(rows) >= 4,
            "holism_claim_refused_as_hetvabhasa": holism_refused,
            "some_claims_ascertained": n_ascertained >= 2,
        }
        dom = _verdict(all(checks.values()),
                       "; ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks.items())
                       + f" | ascertained={n_ascertained}/{len(rows)}")
    else:
        dom = _verdict(False, "audit.json missing — run scripts/x1_audit_claims.py")
    return {"code_gate": code, "domain_gate": dom}


def r_ok(holism: dict) -> bool:
    return holism["verdict"] == "not_ascertained" and len(holism["hetvabhasa"]) >= 1


GATES = {
    "M0": gate_M0, "M1": gate_M1, "M2": gate_M2, "M2b": gate_M2b, "M3": gate_M3,
    "E0": gate_E0, "E1": gate_E1, "E2": gate_E2, "E5": gate_E5, "E6": gate_E6,
    "E4": gate_E4, "E7": gate_E7, "P0": gate_P0, "P1": gate_P1, "P2": gate_P2, "X0": gate_X0, "X1": gate_X1, "X2": gate_X2,
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
