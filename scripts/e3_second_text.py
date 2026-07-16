"""E3 — close one autoresearch cycle: test text-side robustness of the holism finding.

The EFE loop proposed 'e3_second_text_model' (BERT). Agent disposition: BERT is bidirectional,
which is invalid for the incremental cumulative-prefix design (it would see the whole sentence),
so we substitute a second *causal* LM (distilgpt2) — the valid robustness check. This
selector-proposes / agent-disposes split is the prabodha loop pattern, documented here.

Compares cached WavLM speech vs distilgpt2 text on the verb_final contrast; records the outcome
as an EFE observation back into the ledger, then re-proposes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from pranava.autoresearch.loop import LEDGER, LedgerEntry, append_ledger, record_observation
from pranava.experiments.holism import bootstrap_diff
from pranava.experiments.stimuli import generate_stimuli

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "experiments"
FEAT = OUT / "features"

_spec = importlib.util.spec_from_file_location("e2_run", ROOT / "scripts" / "e2_run.py")
e2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2)


def extract_text(stimuli, model_name):
    from pranava.experiments.features import TextFeatureExtractor

    ex = TextFeatureExtractor(layer=3, model_name=model_name)
    return np.stack([ex.positions_matrix(st.text) for st in stimuli])


def main() -> int:
    stimuli = generate_stimuli()
    labels = np.array([s.meaning_label for s in stimuli])
    groups = np.array([s.group for s in stimuli])
    structure = np.array([s.structure for s in stimuli])

    dg_p = FEAT / "distilgpt2.npy"
    dtext = np.load(dg_p) if dg_p.exists() else extract_text(stimuli, "distilgpt2")
    if not dg_p.exists():
        np.save(dg_p, dtext)
    speech = np.load(FEAT / "speech.npy")  # WavLM, from E2

    hi_sp = e2.hi_per_item(speech, labels, groups)
    hi_dt = e2.hi_per_item(dtext, labels, groups)
    vf = structure == "verb_final"
    r = bootstrap_diff(hi_sp[vf], hi_dt[vf], seed=0)

    result = {
        "text_model": "distilgpt2 (causal; substituted for proposed BERT — see docstring)",
        "HI_speech_verb_final": round(float(np.nanmean(hi_sp[vf])), 4),
        "HI_distilgpt2_verb_final": round(float(np.nanmean(hi_dt[vf])), 4),
        "speech_gt_text_verb_final": {
            "effect": round(r.effect, 4), "ci95": [round(r.ci_low, 4), round(r.ci_high, 4)],
            "p_one_sided": round(r.p_one_sided if r.effect >= 0 else 1 - r.p_one_sided, 4),
            "supported": bool(r.supports_direction(True)),
        },
    }
    (OUT / "e3_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    # write an X0-style gate for this cycle, then observe it into the ledger (close the loop)
    supported = result["speech_gt_text_verb_final"]["supported"]
    gate = {
        "milestone": "E3", "code_gate": {"verdict": "pass", "evidence": "E2 pipeline reused"},
        "domain_gate": {"verdict": "pass" if supported else "fail",
                        "evidence": f"speech>distilgpt2 on verb_final CI "
                                    f"{result['speech_gt_text_verb_final']['ci95']} "
                                    f"{'excludes' if supported else 'includes'} 0"},
        "closed": supported,
    }
    gate_p = ROOT / "gates" / "gate_E3.json"
    gate_p.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    record_observation("e3_second_text_model", gate_p, ledger_path=LEDGER)
    append_ledger(LedgerEntry("disposition", "e3_second_text_model",
                              {"note": "BERT→distilgpt2 (causal validity)"}), LEDGER)
    print("\nobservation recorded to ledger; loop can re-propose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
