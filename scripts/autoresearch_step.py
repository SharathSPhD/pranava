"""One step of the Sphoṭa-Bench autoresearch loop (reuses prabodha's EFE selector).

Seeds beliefs from gates already banked this session, then proposes the next experiment.
Prints the ranked menu with EFE / epistemic / pragmatic terms. Persists the proposal.
"""
from __future__ import annotations

import json
from pathlib import Path

from pranava.autoresearch.loop import (
    LEDGER,
    MENU,
    LedgerEntry,
    append_ledger,
    build_selector,
    consumed_ids,
    read_ledger,
    record_observation,
)

ROOT = Path(__file__).resolve().parents[1]


def seed_from_banked_gates() -> None:
    """If the ledger is empty, seed the e2b belief from the real gate_E2.json headroom.

    E2b (HuBERT replication) was actually run and its result is strong; record that as a
    tier-3 observation so the loop reflects reality rather than starting blind.
    """
    if read_ledger(LEDGER):
        return
    gate_e2 = ROOT / "gates" / "gate_E2.json"
    if gate_e2.exists():
        record_observation("e2b_hubert_replication", gate_e2, ledger_path=LEDGER)


def main() -> int:
    seed_from_banked_gates()
    ledger = read_ledger(LEDGER)
    sel = build_selector(ledger)
    done = consumed_ids(ledger)
    remaining = [c for c in MENU if c.id not in done]

    print(f"consumed (run): {sorted(done)}")
    print("\nranked menu (lower EFE = pick first):")
    for p in sel.rank(remaining):
        print(f"  {p.candidate.id:26s} action={p.action.name:7s} "
              f"EFE={p.efe:+.3f} epi={p.epistemic:.3f} prag={p.pragmatic:.3f} "
              f"belief={[round(b,2) for b in p.belief]}")

    prop = sel.select(remaining, budget_gpu_hours=5.0) if remaining else None
    if prop is None:
        print("\nall candidates consumed — menu exhausted.")
        return 0
    print(f"\nPROPOSAL → {prop.candidate.id} via '{prop.action.name}' "
          f"({prop.action.gpu_hours} gpu-h): {prop.candidate.description}")
    append_ledger(
        LedgerEntry("proposal", prop.candidate.id,
                    {"action": prop.action.name, "efe": round(prop.efe, 4),
                     "knobs": prop.candidate.knobs}),
        LEDGER,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
