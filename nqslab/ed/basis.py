"""Configuration bases. Configurations are +-1 arrays; code = sum_i [s_i = +1] 2^i."""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Tuple

import numpy as np


def codes_of(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s)
    N = s.shape[-1]
    return (s == 1).astype(np.int64) @ (1 << np.arange(N, dtype=np.int64))


def configs_of(codes: np.ndarray, N: int) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    return (((codes[..., None] >> np.arange(N)[None, :]) & 1) * 2 - 1).astype(np.int8)


def full_basis(N: int) -> Tuple[np.ndarray, np.ndarray]:
    codes = np.arange(2 ** N, dtype=np.int64)
    return configs_of(codes, N), codes


def sz_basis(N: int, n_up: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """All configurations with n_up up spins (default N // 2) and their codes, sorted by code."""
    n_up = N // 2 if n_up is None else n_up
    combs = list(combinations(range(N), n_up))
    basis = -np.ones((len(combs), N), dtype=np.int8)
    if n_up > 0:
        combs = np.array(combs, dtype=int).reshape(len(combs), n_up)
        basis[np.repeat(np.arange(len(combs)), n_up), combs.ravel()] = 1
    codes = codes_of(basis)
    order = np.argsort(codes)
    return basis[order], codes[order]
