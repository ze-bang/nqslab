"""Exact reference implementations on state vectors (for validating the sampled estimators)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def full_vector(vec: np.ndarray, codes: np.ndarray, N: int) -> np.ndarray:
    out = np.zeros(2 ** N, dtype=complex)
    out[codes] = vec
    return out


def _bipartition(vec, codes, N, mask):
    full = full_vector(vec, codes, N)
    mask = np.asarray(mask, dtype=bool)
    A = np.where(mask)[0]; B = np.where(~mask)[0]
    idx = np.arange(2 ** N)
    bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
    ca = bits[:, A] @ (1 << np.arange(len(A))); cb = bits[:, B] @ (1 << np.arange(len(B)))
    M = np.zeros((2 ** len(A), 2 ** len(B)), dtype=complex)
    M[ca, cb] = full
    return M


def renyi2_exact(vec, codes, N, mask) -> float:
    sv = np.linalg.svd(_bipartition(vec, codes, N, mask), compute_uv=False)
    p = sv ** 2 / np.sum(sv ** 2)
    return float(-np.log(np.sum(p ** 2)))


def entanglement_entropy_exact(vec, codes, N, mask) -> float:
    sv = np.linalg.svd(_bipartition(vec, codes, N, mask), compute_uv=False)
    p = sv ** 2 / np.sum(sv ** 2); p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def rdm_exact(vec, codes, N, mask) -> np.ndarray:
    M = _bipartition(vec, codes, N, mask)
    return M @ M.conj().T


def plaquette_berry_phase_exact(corner_states: Sequence[np.ndarray]) -> float:
    """corner_states[k]: (dim, d) orthonormal multiplet at corner k. F = arg det prod U."""
    P = np.eye(corner_states[0].shape[1], dtype=complex)
    for k in range(4):
        P = P @ (corner_states[k].conj().T @ corner_states[(k + 1) % 4])
    return float(np.angle(np.linalg.det(P)))


def chern_number_exact(states_on_grid) -> int:
    """states_on_grid[i][j]: (dim, d) multiplet at twist grid point (i, j) (periodic grid)."""
    n1 = len(states_on_grid); n2 = len(states_on_grid[0])
    total = 0.0
    for i in range(n1):
        for j in range(n2):
            corners = [states_on_grid[i][j], states_on_grid[(i + 1) % n1][j],
                       states_on_grid[(i + 1) % n1][(j + 1) % n2], states_on_grid[i][(j + 1) % n2]]
            total += plaquette_berry_phase_exact(corners)
    return int(round(total / (2 * np.pi)))
