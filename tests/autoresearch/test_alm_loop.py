"""TDD: the ALM-improvement loop scores iterations by CER gain (host-runnable)."""
import pytest

pytest.importorskip("prabodha.efe.agent")

from pranava.autoresearch.alm_loop import MENU, cer_to_tier  # noqa: E402


def test_big_improvement_is_top_tier():
    assert cer_to_tier(0.77, 0.55) == 3   # −0.22 → strong
    assert cer_to_tier(0.60, 0.54) == 3   # −0.06 → strong


def test_small_improvement_is_mid_tier():
    assert cer_to_tier(0.60, 0.585) == 2  # −0.015 → modest


def test_neutral_is_low_tier():
    assert cer_to_tier(0.60, 0.605) == 1  # ~neutral, still informative


def test_regression_is_zero():
    assert cer_to_tier(0.55, 0.62) == 0   # worse → prune


def test_no_result_is_zero():
    assert cer_to_tier(0.55, None) == 0


def test_menu_has_gpu_placed_candidates():
    ids = {c.id for c in MENU}
    assert "lora_r8_200m" in ids and "lora_1b_5090" in ids and "real_speech_data" in ids
    # candidates carry a GPU placement and an integer value hint
    for c in MENU:
        assert isinstance(c.prior_value_hint, int) and 0 <= c.prior_value_hint <= 3
