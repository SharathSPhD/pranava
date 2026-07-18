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

# The menu of platform improvements. prior_value_hint ∈ {0..3} (negligible..high expected value).
# Expanded 2026-07-17: multilingual, app, instruction-tuning/RLAIF, and NSM-architecture candidates —
# pranava as a platform + research testbed, not a single Sanskrit model.
MENU: list[Candidate] = [
    Candidate("lora_r8_200m", "LoRA r=8 on the 200M core (adapt, don't just project)",
              {"where": "gb10", "r": 8}, 3),
    Candidate("lora_1b_5090", "LoRA on the 1.13B Megatron core (RTX 5090)",
              {"where": "5090", "r": 8}, 3),
    Candidate("real_speech_data", "Native-Sanskrit speech corpus (indic-parler-tts)",
              {"data": "native_sa"}, 3),
    Candidate("multilingual_english", "Multilingual ALM: add REAL English speech (LibriSpeech/CV)",
              {"lang": "en", "data": "real"}, 3),
    Candidate("multilingual_indic", "Add Hindi/other Indic real speech (cross-lingual transfer)",
              {"lang": "hi"}, 2),
    Candidate("instruction_tuning", "SFT the ALM to follow spoken instructions (usable product)",
              {"stage": "sft"}, 3),
    Candidate("rlaif", "RLAIF: align with an AI reward model (NeMo/Nemotron reward)",
              {"stage": "rlaif"}, 2),
    Candidate("nsm_layer_c", "Inline Navya-Nyaya (pramana) validation over ALM claims (NSM Layer C)",
              {"nsm": "C"}, 2),
    Candidate("vercel_app", "Serve the ALM + Vercel/Supabase app (make it usable)",
              {"deploy": "vercel"}, 3),
    Candidate("lens_guided_decoding", "Steer the sphoṭa layer during decode toward meaning",
              {"steer_layer": 13}, 2),
    Candidate("more_epochs", "Longer projector+LoRA schedule (20+ epochs)", {"epochs": 20}, 1),
    # ---- 2026-07-18 campaign (post benchmark correction: fair free-decode cer_norm is the target) ----
    Candidate("xl_corpus", "16× corpus: TTS the full PSALM paninian_v1 fixture (~10k clips)",
              {"data": "speech_corpus_indic_xl"}, 3),
    Candidate("xl_200m_eos", "Retrain 200M projector+LoRA on XL corpus, EOS-weighted, fair eval",
              {"where": "gb10", "corpus": "xl", "eos_weight": 3.0}, 3),
    Candidate("xl_1b_lora", "1.13B + Megatron-LoRA (mlp+attn) on XL corpus (RTX 5090)",
              {"where": "5090", "corpus": "xl", "r": 16}, 3),
    Candidate("steering_uptake", "Port prabodha write-steering to the sphoṭa band; readback-verified uptake",
              {"band": [21, 23], "reuse": "prabodha.steering"}, 2),
    Candidate("nyaya_guardrail", "pramāṇa/nyāya legality gate inside decoding (steer or re-decode on failure)",
              {"reuse": "pramana.validators"}, 2),
    Candidate("fusion_v2", "Causal cross-modal integration metric for the sphoṭa-lens (replace CKA)",
              {"method": "causal_mediation"}, 2),
]


def leaderboard_cer(model_substr: str = "ours") -> float | None:
    if not LEADERBOARD.exists():
        return None
    b = json.loads(LEADERBOARD.read_text())
    for r in b.get("leaderboard", []):
        if model_substr in r.get("model", "") and r.get("cer") is not None:
            return float(r["cer"])
    return None


FAIR_BOARD = ROOT / "data" / "benchmark" / "alm_vs_alm.json"


def fair_cer_norm(model_substr: str = "ours") -> float | None:
    """Best (lowest) transliteration-normalized CER for our model on the FAIR multi-ALM leaderboard.

    This is the campaign objective post benchmark-correction (2026-07-18): free decode, no
    gold-length oracle, scheme-neutral metric — the number that must beat the open generalists
    (Voxtral 0.187) for an honest 'top of the leaderboard' claim."""
    if not FAIR_BOARD.exists():
        return None
    b = json.loads(FAIR_BOARD.read_text())
    ours = [r.get("cer_norm") for r in b.get("leaderboard", [])
            if model_substr in r.get("model", "") and r.get("cer_norm") is not None
            and "oracle" not in r.get("model", "")]  # oracle-capped rows don't count
    return min(ours) if ours else None


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
            consumed.add(e["candidate_id"])
            tier = e.get("primary_tier")
            if tier is None:  # legacy/free-form entries (e.g. milestone notes) count as consumed only
                continue
            sel.update(e["candidate_id"], Observation(primary_tier=int(tier)))
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
