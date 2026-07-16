"""TDD: the Sphoṭa-Bench autoresearch loop (reuses prabodha's EFE selector)."""
import json

import pytest

prabodha = pytest.importorskip("prabodha.efe.agent", reason="prabodha not importable")

from pranava.autoresearch.loop import (  # noqa: E402
    MENU,
    LedgerEntry,
    build_selector,
    consumed_ids,
    gate_to_observation,
    propose_next,
    record_observation,
)


def _write_gate(tmp_path, code, dom, evidence=""):
    p = tmp_path / "gate.json"
    p.write_text(json.dumps({
        "code_gate": {"verdict": code, "evidence": ""},
        "domain_gate": {"verdict": dom, "evidence": evidence},
    }))
    return p


def test_gate_pass_maps_to_high_tier(tmp_path):
    g = _write_gate(tmp_path, "pass", "pass", "effect CI [0.106, 0.361] excludes 0")
    assert gate_to_observation(g).primary_tier == 3


def test_gate_pass_plain_is_tier2(tmp_path):
    g = _write_gate(tmp_path, "pass", "pass", "all checks ok")
    assert gate_to_observation(g).primary_tier == 2


def test_gate_domain_fail_is_near_miss(tmp_path):
    g = _write_gate(tmp_path, "pass", "fail")
    assert gate_to_observation(g).primary_tier == 1


def test_gate_code_fail_is_zero(tmp_path):
    g = _write_gate(tmp_path, "fail", "fail")
    assert gate_to_observation(g).primary_tier == 0


def test_propose_returns_a_menu_candidate(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    prop = propose_next(ledger_path=ledger)
    assert prop is not None
    assert prop.candidate.id in {c.id for c in MENU}
    assert prop.action.name in {"smoke", "partial", "full"}


def test_consumed_candidates_are_excluded(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    first = propose_next(ledger_path=ledger)
    # record an observation for the first proposal → it must not be proposed again
    g = _write_gate(tmp_path, "pass", "pass", "excludes 0")
    record_observation(first.candidate.id, g, ledger_path=ledger)
    second = propose_next(ledger_path=ledger)
    assert second is None or second.candidate.id != first.candidate.id


def test_belief_update_from_ledger_is_reentrant(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    g = _write_gate(tmp_path, "pass", "pass", "excludes 0")
    record_observation("e2b_hubert_replication", g, ledger_path=ledger)
    from pranava.autoresearch.loop import read_ledger

    sel = build_selector(read_ledger(ledger))
    cand = next(c for c in MENU if c.id == "e2b_hubert_replication")
    belief = sel.belief(cand)
    # a tier-3 observation should push belief mass toward higher value levels
    assert belief[-1] > belief[0]


def test_loop_terminates_when_all_consumed(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    g = _write_gate(tmp_path, "pass", "pass", "excludes 0")
    for c in MENU:
        record_observation(c.id, g, ledger_path=ledger)
    assert propose_next(ledger_path=ledger) is None


def test_ledger_entry_roundtrip():
    e = LedgerEntry("observation", "x", {"primary_tier": 2})
    d = json.loads(e.to_json())
    assert d["kind"] == "observation" and d["candidate_id"] == "x" and d["primary_tier"] == 2
