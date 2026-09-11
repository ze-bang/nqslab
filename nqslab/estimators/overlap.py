"""Overlap estimators. All log_psi arguments are callables s (B, N) -> complex (B,)."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _mean_err(x: np.ndarray) -> Tuple[complex, float]:
    x = np.asarray(x)
    return complex(np.mean(x)), float(np.std(x) / np.sqrt(len(x)))


def overlap_ratio(log_psi_a: Callable, log_psi_b: Callable, samples_a: np.ndarray) -> Tuple[complex, float]:
    """<a|b>/<a|a> = E_{|a|^2}[ psi_b / psi_a ]  (carries the relative phase)."""
    r = np.exp(np.asarray(log_psi_b(samples_a)) - np.asarray(log_psi_a(samples_a)))
    return _mean_err(r)


def fidelity(log_psi_a: Callable, log_psi_b: Callable, samples_a: np.ndarray, samples_b: np.ndarray) -> Tuple[float, float]:
    """|<a|b>|^2 / (<a|a><b|b>) = E_a[psi_b/psi_a] E_b[psi_a/psi_b]."""
    ra, ea = overlap_ratio(log_psi_a, log_psi_b, samples_a)
    rb, eb = overlap_ratio(log_psi_b, log_psi_a, samples_b)
    f = (ra * rb).real
    err = abs(f) * np.sqrt((ea / abs(ra)) ** 2 + (eb / abs(rb)) ** 2) if abs(ra) > 0 and abs(rb) > 0 else np.nan
    return float(f), float(err)


def orthonormalize_pair(log_psi_1: Callable, log_psi_2: Callable, samples_1: np.ndarray, samples_2: np.ndarray):
    """c = <1|2>/<1|1>, so that psi_2'(s) = psi_2(s) - c psi_1(s) is orthogonal to psi_1."""
    c, _ = overlap_ratio(log_psi_1, log_psi_2, samples_1)
    return c
