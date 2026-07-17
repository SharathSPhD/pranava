"""One step of the continuous ALM-improvement loop: seed completed iterations, propose the next.

Seeds the ledger from what has actually been run (LoRA r=8 moved CER 0.774→0.548), then proposes
the next improvement to run, ranked by expected free energy against the live SOTA leaderboard.
"""
from __future__ import annotations

import json

from pranava.autoresearch.alm_loop import (
    LEDGER,
    MENU,
    build_selector,
    leaderboard_cer,
    propose_next,
    record_run,
)
from pranava.autoresearch.loop import LedgerEntry, append_ledger, read_ledger


def seed():
    """Record already-run iterations once (idempotent)."""
    done = {e["candidate_id"] for e in read_ledger(LEDGER) if e.get("kind") == "observation"}
    if "lora_r8_200m" not in done:
        record_run("lora_r8_200m", prev_cer=0.7745, new_cer=0.5475,
                   note="LoRA r=8 on 200M core; beats 1B projector-only")
        append_ledger(LedgerEntry("disposition", "lora_r8_200m",
                                  {"note": "current best; now the SOTA-leaderboard #1 (0.546)"}), LEDGER)


def main() -> int:
    seed()
    sel, consumed = build_selector()
    remaining = [c for c in MENU if c.id not in consumed]
    print(f"live leaderboard CER (ours): {leaderboard_cer()}")
    print(f"consumed: {sorted(consumed)}\n")
    print("ranked next improvements (lower EFE first):")
    for p in sel.rank(remaining):
        print(f"  {p.candidate.id:22s} action={p.action.name:7s} EFE={p.efe:+.3f} "
              f"prior={p.candidate.prior_value_hint}  {p.candidate.description[:52]}")
    prop = propose_next()
    if prop:
        print(f"\nPROPOSAL → {prop.candidate.id} ({prop.action.name}): {prop.candidate.description}")
        append_ledger(LedgerEntry("proposal", prop.candidate.id,
                                  {"action": prop.action.name, "knobs": prop.candidate.knobs}), LEDGER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
