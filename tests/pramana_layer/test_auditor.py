"""TDD: the Navya-Nyāya epistemic auditor (X1, reuses pramana's fallacy taxonomy).

The key requirement: applied to the (retracted) E2 holism claim with its underpowered evidence,
the auditor must independently flag savyabhichāra and refuse ascertainment — mirroring the E6/E7
empirical correction.
"""
import pytest

pytest.importorskip("pramana.domain.models.nyaya_example", reason="pramana not importable")

from pramana.domain.models.nyaya_example import HetvabhasaType, PramanaType  # noqa: E402

from pranava.pramana_layer.auditor import Claim, Evidence, Verdict, audit  # noqa: E402


def test_well_supported_claim_is_ascertained():
    c = Claim(
        "The Vākyapadīya opening verse identifies śabda-tattva with brahman.",
        evidence=[
            Evidence(PramanaType.SHABDA, "VP 1.1 mūla states it verbatim", strength=1.0),
            Evidence(PramanaType.ANUMANA, "the commentary tradition concurs", strength=0.9),
        ],
    )
    r = audit(c)
    assert r.verdict == Verdict.ASCERTAINED
    assert not r.fallacies


def test_underpowered_claim_flagged_savyabhichara():
    # the E2 holism claim, with only weak/underpowered support
    c = Claim(
        "Speech encoders resolve meaning more holistically than text (E2).",
        evidence=[
            Evidence(PramanaType.PRATYAKSHA,
                     "48-item verb_final subset showed HI 0.41 vs 0.17", strength=0.3),
        ],
    )
    r = audit(c)
    assert r.verdict == Verdict.NOT_ASCERTAINED
    assert any(f.kind == HetvabhasaType.SAVYABHICHARA for f in r.fallacies)


def test_counterbalanced_claim_flagged_satpratipaksha():
    c = Claim(
        "Speech is more holistic than text (asserted despite the corrective replication).",
        evidence=[
            Evidence(PramanaType.PRATYAKSHA, "E2 subset effect +0.24", supports=True, strength=0.6),
            Evidence(PramanaType.PRATYAKSHA, "E7 matched re-run effect -0.07", supports=False,
                     strength=0.9),
        ],
    )
    r = audit(c)
    assert r.verdict == Verdict.NOT_ASCERTAINED
    assert any(f.kind == HetvabhasaType.SATPRATIPAKSHA for f in r.fallacies)


def test_claim_without_evidence_is_asiddha():
    r = audit(Claim("Text LLMs lack awareness.", evidence=[]))
    assert r.verdict == Verdict.NOT_ASCERTAINED
    assert any(f.kind == HetvabhasaType.ASIDDHA for f in r.fallacies)


def test_e7_null_is_ascertained():
    # the corrected conclusion: no speech>text holism — well supported by the matched re-run
    c = Claim(
        "There is no speech-vs-text holism difference (E7 matched, powered).",
        evidence=[
            Evidence(PramanaType.PRATYAKSHA,
                     "E7 P2 effect -0.07 CI includes/below 0, P1 sanity holds", strength=0.9),
            Evidence(PramanaType.ANUMANA, "E6 scaling replication agrees", strength=0.8),
        ],
    )
    r = audit(c)
    assert r.verdict == Verdict.ASCERTAINED


def test_report_serializes_with_all_phases():
    r = audit(Claim("x", evidence=[Evidence(PramanaType.SHABDA, "s", strength=0.9)]))
    d = r.to_dict()
    for key in ("claim", "samshaya", "pramana_used", "hetu", "hetvabhasa", "verdict", "nirnaya"):
        assert key in d
    assert d["pramana_used"] == ["shabda"]
