"""Many-body Chern number from one plaquette of boundary twists.

Corners k = 0..3 in the order (0,0), (d,0), (d,d), (0,d). For each corner a list of d state
functions (the multiplet) and samples from each of them. The link matrix
U_ab(k -> k') = <psi_a^k | psi_b^k'> / <psi_a^k|psi_a^k> is estimated with samples of psi_a^k, and
F = arg det[ U(0->1) U(1->2) U(2->3) U(3->0) ] is gauge invariant.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

from .overlap import overlap_ratio


def link_matrix(states_k: Sequence[Callable], states_kp: Sequence[Callable], samples_k: Sequence[np.ndarray]) -> np.ndarray:
    d = len(states_k)
    U = np.zeros((d, d), dtype=complex)
    for a in range(d):
        for b in range(d):
            U[a, b], _ = overlap_ratio(states_k[a], states_kp[b], samples_k[a])
    return U


def plaquette_berry_phase(corners: Sequence[Sequence[Callable]], samples: Sequence[Sequence[np.ndarray]]) -> Tuple[float, np.ndarray]:
    """(F, product matrix). corners[k][a] is log psi of state a at corner k."""
    P = np.eye(len(corners[0]), dtype=complex)
    for k in range(4):
        P = P @ link_matrix(corners[k], corners[(k + 1) % 4], samples[k])
    return float(np.angle(np.linalg.det(P))), P


def chern_from_corners(corners, samples, delta: float) -> Tuple[float, float]:
    """Single-plaquette Chern estimate C ~ F / delta^2 (the plaquette covers (delta/2pi)^2 of the twist torus)."""
    F, _ = plaquette_berry_phase(corners, samples)
    return F / delta ** 2, F
