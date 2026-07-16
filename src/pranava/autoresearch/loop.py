"""Sphoṭa-Bench autoresearch loop — reuses prabodha's EFE selector.

The selector proposes the next experiment to run (by expected information gain per
GPU-hour); pranava's dual-verdict gates become tiered observations that update beliefs;
a JSONL ledger makes the beliefs re-entrant across sessions. This is the pranava
instantiation of prabodha's L5 autoresearch loop (prabodha.efe), applied to Pillar II.

Reuses (not copies): ``prabodha.efe.agent`` (made importable via a .pth in the venv).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from prabodha.efe.agent import Candidate, EFESelector, Observation

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "research" / "efe_ledger.jsonl"


def gate_to_observation(gate_path: str | Path) -> Observation:
    """Map a pranava dual-verdict gate JSON → EFE Observation tier.

    pranava gates carry {code_gate:{verdict}, domain_gate:{verdict, evidence}}.
    3 = both pass and domain evidence records a numeric headroom marker,
    2 = both pass, 1 = code passes but domain fails (near miss), 0 = code fails.
    """
    g = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    code = g.get("code_gate", {}).get("verdict")
    dom = g.get("domain_gate", {}).get("verdict")
    if code == "pass" and dom == "pass":
        ev = str(g.get("domain_gate", {}).get("evidence", ""))
        # headroom heuristic: an effect/coverage/CI marker well clear of its floor
        tier = 3 if ("CI [0.1" in ev or "coverage=0.4" in ev or "excludes" in ev) else 2
    elif code == "pass":
        tier = 1
    else:
        tier = 0
    return Observation(primary_tier=tier)


# The Sphoṭa-Bench experiment menu. knobs are passed to dispatch; prior_value_hint is a discrete
# value-level index 0..3 (negligible/low/moderate/high) per prabodha's EFE prior.
MENU: list[Candidate] = [
    Candidate(
        id="e2b_hubert_replication",
        description="Replicate verb-final holism with HuBERT (second speech encoder)",
        knobs={"speech_model": "facebook/hubert-base-ls960"},
        prior_value_hint=3,
    ),
    Candidate(
        id="e3_second_text_model",
        description="Add BERT as a second text model to test text-side robustness",
        knobs={"text_model": "bert-base-uncased"},
        prior_value_hint=2,
    ),
    Candidate(
        id="e4_layer_sweep",
        description="Sweep speech/text layers to locate where holism peaks",
        knobs={"layers": "0..12"},
        prior_value_hint=1,
    ),
    Candidate(
        id="e5_prosody_manipulation",
        description="Prosody-carrying items (sarcasm/urgency) — pragmatic content speech-only",
        knobs={"stimulus": "prosody"},
        prior_value_hint=2,
    ),
    Candidate(
        id="e6_scale_verbfinal",
        description="Scale verb-final stimuli (more items) to tighten the CI on the key effect",
        knobs={"stimulus": "verb_final_x4"},
        prior_value_hint=2,
    ),
]


@dataclass(slots=True)
class LedgerEntry:
    kind: str  # "proposal" | "observation"
    candidate_id: str
    payload: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "candidate_id": self.candidate_id, **self.payload},
                          ensure_ascii=False)


def read_ledger(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def append_ledger(entry: LedgerEntry, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.to_json() + "\n")


def consumed_ids(ledger: list[dict]) -> set[str]:
    """Candidates that already have an observation (been run) → excluded from proposal."""
    return {e["candidate_id"] for e in ledger if e.get("kind") == "observation"}


def build_selector(ledger: list[dict]) -> EFESelector:
    """Rebuild beliefs by replaying observations from the ledger (re-entrant)."""
    sel = EFESelector()
    for e in ledger:
        if e.get("kind") == "observation":
            sel.update(e["candidate_id"], Observation(primary_tier=int(e["primary_tier"])))
    return sel


def propose_next(
    menu: list[Candidate] = MENU,
    ledger_path: Path = LEDGER,
    budget_gpu_hours: float = 5.0,
):
    """Return the top EFE proposal over not-yet-run candidates, or None if all consumed."""
    ledger = read_ledger(ledger_path)
    sel = build_selector(ledger)
    done = consumed_ids(ledger)
    remaining = [c for c in menu if c.id not in done]
    if not remaining:
        return None
    return sel.select(remaining, budget_gpu_hours=budget_gpu_hours)


def record_observation(candidate_id: str, gate_path: str | Path,
                       ledger_path: Path = LEDGER) -> Observation:
    """Observe a gate outcome for a run candidate and persist it to the ledger."""
    obs = gate_to_observation(gate_path)
    append_ledger(
        LedgerEntry("observation", candidate_id,
                    {"primary_tier": obs.primary_tier, "gate": str(gate_path)}),
        ledger_path,
    )
    return obs
