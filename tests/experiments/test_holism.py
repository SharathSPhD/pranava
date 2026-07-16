"""TDD: Holism Index + bootstrap on synthetic curves (no models needed)."""
import numpy as np

from pranava.experiments.holism import (
    POSITIONS,
    BootstrapResult,
    bootstrap_diff,
    holism_index,
    holm_correction,
)


def test_late_flash_curve_has_high_hi():
    # flat then jumps at the very end → holistic
    curve = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12, 0.15, 0.9])
    assert holism_index(curve) > 0.8


def test_linear_accrual_curve_has_low_hi():
    curve = np.linspace(0.1, 0.9, len(POSITIONS))
    # last 20% (one step of ten) contributes ~1/9 of total gain
    assert holism_index(curve) < 0.25  # last 20% = 2/9 steps of a linear ramp


def test_flat_curve_is_nan():
    curve = np.full(len(POSITIONS), 0.3)
    assert np.isnan(holism_index(curve))


def test_hi_clipped_to_unit_interval():
    curve = np.array([0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.9])
    hi = holism_index(curve)
    assert 0.0 <= hi <= 1.0


def test_bootstrap_detects_positive_effect():
    rng = np.random.default_rng(1)
    a = rng.normal(0.7, 0.1, 200)
    b = rng.normal(0.3, 0.1, 200)
    r = bootstrap_diff(a, b, n_boot=1000, seed=0)
    assert isinstance(r, BootstrapResult)
    assert r.effect > 0
    assert r.excludes_zero
    assert r.supports_direction(positive=True)


def test_bootstrap_null_includes_zero():
    rng = np.random.default_rng(2)
    a = rng.normal(0.5, 0.1, 200)
    b = rng.normal(0.5, 0.1, 200)
    r = bootstrap_diff(a, b, n_boot=1000, seed=0)
    assert not r.excludes_zero


def test_bootstrap_deterministic():
    a = np.linspace(0, 1, 50)
    b = np.linspace(0.2, 0.8, 50)
    r1 = bootstrap_diff(a, b, n_boot=500, seed=7)
    r2 = bootstrap_diff(a, b, n_boot=500, seed=7)
    assert (r1.ci_low, r1.ci_high, r1.effect) == (r2.ci_low, r2.ci_high, r2.effect)


def test_bootstrap_drops_nans():
    a = np.array([0.5, 0.6, np.nan, 0.7])
    b = np.array([0.1, np.nan, 0.2, 0.3])
    r = bootstrap_diff(a, b, n_boot=200, seed=0)
    assert r.n == 6  # 3 + 3 valid after dropping one nan each


def test_holm_correction():
    # one clearly significant, one clearly not
    rej = holm_correction([0.001, 0.9])
    assert rej == [True, False]
    # both significant
    assert holm_correction([0.001, 0.002]) == [True, True]
