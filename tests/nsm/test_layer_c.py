"""TDD for NSM Layer C — the pramāṇa validation gate over ALM outputs (host-testable, pure)."""
import pytest

pytest.importorskip("pramana.domain.models.nyaya_example", reason="pramana not importable")

from pranava.nsm.layer_c import (  # noqa: E402
    is_ascertained,
    repetition_ratio,
    romanization_legality,
    validate_output,
)
from pranava.pramana_layer.auditor import Verdict  # noqa: E402


def test_clean_confident_output_is_ascertained():
    # a legal, non-repeating, high-confidence Sanskrit utterance clears every phase
    r = validate_output("bAlAH vidyayA pustakam KAdeyuH", confidence=0.85)
    assert r.verdict is Verdict.ASCERTAINED


def test_looping_output_is_rejected():
    # the "vane vane vane" failure mode: repetition overturns support (satpratipakṣa)
    assert is_ascertained("vane vane vane vane", confidence=0.8) is False


def test_low_confidence_is_not_ascertained():
    # weak pratyakṣa alone → savyabhichāra (erratic reason), refuse ascertainment
    r = validate_output("naraH gfham paWet", confidence=0.2)
    assert r.verdict is not Verdict.ASCERTAINED


def test_signal_helpers():
    assert repetition_ratio("a a a") == 1.0
    assert repetition_ratio("a b c") == 0.0
    assert romanization_legality("abc def") == 1.0
    assert romanization_legality("ab\x00\x01") == 0.5  # 2 legal of 4 chars
