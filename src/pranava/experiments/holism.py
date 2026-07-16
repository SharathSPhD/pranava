"""Holism Index + bootstrap statistics for the H-HOLISM experiment (E2).

Kept free of any model/GPU dependency so the metric and inference are unit-testable
on synthetic curves. See research/prereg/H-HOLISM.md for the frozen definitions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSITIONS = np.round(np.arange(0.1, 1.0001, 0.1), 4)  # 0.1 .. 1.0
_EPS = 1e-9


def holism_index(curve: np.ndarray, positions: np.ndarray = POSITIONS) -> float:
    """Fraction of total decodability gain achieved in the last 20% of the utterance.

    curve[i] = decodability at positions[i]. HI = (a(1.0)-a(0.8)) / (a(1.0)-a(0.1)).
    Clipped to [0, 1]. If total gain is ~0 (flat curve), HI is undefined → returns nan.
    """
    curve = np.asarray(curve, dtype=float)
    a_end = curve[-1]
    # nearest positions to 0.8 and 0.1
    i08 = int(np.argmin(np.abs(positions - 0.8)))
    i01 = int(np.argmin(np.abs(positions - 0.1)))
    total = a_end - curve[i01]
    if abs(total) < 1e-6:
        return float("nan")
    hi = (a_end - curve[i08]) / (total + _EPS)
    return float(np.clip(hi, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    effect: float  # observed mean difference (a - b)
    ci_low: float
    ci_high: float
    n: int
    p_one_sided: float = float("nan")  # P(bootstrap diff <= 0) when effect>0 (else >=0)

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def supports_direction(self, positive: bool = True) -> bool:
        return (self.ci_low > 0) if positive else (self.ci_high < 0)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0, ci: float = 95.0
) -> BootstrapResult:
    """Bootstrap CI for the difference of means of two independent samples.

    NaNs are dropped (logged by the caller via counts). Deterministic given seed.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0)
    rng = _rng(seed)
    obs = float(np.mean(a) - np.mean(b))
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.mean(sa) - np.mean(sb)
    lo = float(np.percentile(diffs, (100 - ci) / 2))
    hi = float(np.percentile(diffs, 100 - (100 - ci) / 2))
    # one-sided bootstrap p-value in the direction of the observed effect
    if obs >= 0:
        p = float(np.mean(diffs <= 0.0))
    else:
        p = float(np.mean(diffs >= 0.0))
    return BootstrapResult(effect=obs, ci_low=lo, ci_high=hi, n=len(a) + len(b), p_one_sided=p)


def holm_correction(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm–Bonferroni: return per-test reject decisions (True = reject null)."""
    m = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvals[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject
