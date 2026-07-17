"""Continuous ALM-improvement loop — EFE-driven iteration toward robust outcomes.

Reuses prabodha's Expected-Free-Energy selector (via pranava.autoresearch.loop) to propose the next
Śabda-ALM improvement, scored by expected gain per GPU-hour. The objective is NOT falsification but
iteration toward a more robust, SOTA-beating model: each run is scored by how much it moves the live
SOTA leaderboard (data/benchmark/sota_leaderboard.json). Completed iterations are replayed from the
ledger so the loop is re-entrant and never repeats consumed work.
"""
from __future__ import annotations

import json
from pathlib import Path

from prabodha.efe.agent import Candidate, EFESelector, Observation

from pranava.autoresearch.loop import LedgerEntry, append_ledger, read_ledger

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "research" / "alm_efe_ledger.jsonl"
LEADERBOARD = ROOT / "data" / "benchmark" / "sota_leaderboard.json"

# The menu of ALM improvements. prior_value_hint ∈ {0..3} (negligible..high expected value).
MENU: list[Candidate] = [
    Candidate("lora_r8_200m", "LoRA r=8 on the 200M core (adapt, don't just project)",
              {"where": "gb10", "r": 8}, 3),
    Candidate("lora_r16_200m", "Higher-rank LoRA r=16 on the 200M core",
              {"where": "gb10", "r": 16}, 2),
    Candidate("lora_1b_5090", "LoRA on the 1.13B Megatron core (RTX 5090)",
              {"where": "5090", "r": 8}, 3),
    Candidate("more_epochs", "Longer projector+LoRA schedule (20 epochs)",
              {"epochs": 20}, 2),
    Candidate("real_speech_data", "Replace synthetic TTS with real Indic/Sanskrit speech",
              {"data": "real"}, 3),
    Candidate("lens_guided_decoding", "Steer the sphoṭa layer (13) during decode toward meaning",
              {"steer_layer": 13}, 2),
    Candidate("larger_projector", "Deeper Q-Former projector (more madhyamā capacity)",
              {"projector": "qformer"}, 1),
]


def leaderboard_cer(model_substr: str = "ours") -> float | None:
    if not LEADERBOARD.exists():
        return None
    b = json.loads(LEADERBOARD.read_text())
    for r in b.get("leaderboard", []):
        if model_substr in r.get("model", "") and r.get("cer") is not None:
            return float(r["cer"])
    return None


def cer_to_tier(prev_cer: float | None, new_cer: float | None) -> int:
    """Map a CER change into an EFE observation tier (higher = more valuable improvement)."""
    if new_cer is None:
        return 0
    if prev_cer is None:
        return 2 if new_cer < 0.7 else 1
    delta = prev_cer - new_cer  # positive = improved
    if delta >= 0.05:
        return 3
    if delta > 0.0:
        return 2
    if delta > -0.02:
        return 1  # roughly neutral — still informative
    return 0  # regression


def build_selector() -> tuple[EFESelector, set[str]]:
    ledger = read_ledger(LEDGER)
    sel = EFESelector()
    consumed = set()
    for e in ledger:
        if e.get("kind") == "observation":
            sel.update(e["candidate_id"], Observation(primary_tier=int(e["primary_tier"])))
            consumed.add(e["candidate_id"])
    return sel, consumed


def propose_next(budget_gpu_hours: float = 5.0):
    sel, consumed = build_selector()
    remaining = [c for c in MENU if c.id not in consumed]
    if not remaining:
        return None
    return sel.select(remaining, budget_gpu_hours=budget_gpu_hours)


def record_run(candidate_id: str, prev_cer: float | None, new_cer: float | None, note: str = "") -> int:
    tier = cer_to_tier(prev_cer, new_cer)
    append_ledger(LedgerEntry("observation", candidate_id,
                              {"primary_tier": tier, "prev_cer": prev_cer, "new_cer": new_cer,
                               "note": note}), LEDGER)
    return tier
