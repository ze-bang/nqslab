"""Minimally entangled states and the modular S matrix.

Given an orthonormal doublet (psi_1, psi_2) as log-amplitude callables, the superposition
Xi_c = c_1 psi_1 + c_2 psi_2 has log Xi = logaddexp(log c_1 + log psi_1, log c_2 + log psi_2), and
S_2 of a cut region is minimised over c on the Bloch sphere. Sampling |Xi|^2 requires a sampler for
an arbitrary numpy log-amplitude; :func:`numpy_exchange_sampler` is provided.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .renyi import renyi2


def superposition(log_psi_1: Callable, log_psi_2: Callable, c: np.ndarray) -> Callable:
    c1, c2 = complex(c[0]), complex(c[1])
    def f(s):
        a = np.asarray(log_psi_1(s)) + np.log(c1 + 0j); b = np.asarray(log_psi_2(s)) + np.log(c2 + 0j)
        m = np.maximum(a.real, b.real)
        return m + np.log(np.exp(a - m) + np.exp(b - m))
    return f


def bloch(theta: float, phi: float) -> np.ndarray:
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def numpy_exchange_sampler(log_psi: Callable, pairs: np.ndarray, N: int, n_chains: int, n_sweeps: int, rng,
                           s0: Optional[np.ndarray] = None, p_long: float = 0.2, thin: int = 1, n_burn: int = 10) -> np.ndarray:
    """Metropolis exchange sampler in numpy for any callable log_psi (pairs: candidate site pairs, e.g. nn bonds)."""
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if s0 is None:
        s0 = -np.ones((n_chains, N), dtype=int)
        for c in range(n_chains):
            s0[c, rng.choice(N, N // 2, replace=False)] = 1
    s = s0.copy(); lp = np.asarray(log_psi(s))
    out = []; rows = np.arange(n_chains)
    for sweep in range(n_burn + n_sweeps):
        for _ in range(N):
            use_long = rng.random(n_chains) < p_long
            b = rng.integers(0, len(pairs), n_chains)
            i = np.where(use_long, rng.integers(0, N, n_chains), pairs[b, 0])
            j = np.where(use_long, (i + rng.integers(1, N, n_chains)) % N, pairs[b, 1])
            differ = s[rows, i] != s[rows, j]
            sp = s.copy(); sp[rows, i] = s[rows, j]; sp[rows, j] = s[rows, i]
            lpp = np.asarray(log_psi(sp))
            acc = differ & (rng.random(n_chains) < np.exp(2 * (lpp - lp).real))
            s[acc] = sp[acc]; lp[acc] = lpp[acc]
        if sweep >= n_burn and (sweep - n_burn) % thin == 0:
            out.append(s.copy())
    return np.concatenate(out, axis=0)


def mes_search(log_psi_1: Callable, log_psi_2: Callable, mask: np.ndarray, sampler: Callable, grid: int = 6,
               refine: bool = True):
    """Minimise S_2(mask) over the Bloch sphere; ``sampler(f)`` returns samples of |f|^2.
    Returns (c_min, S2_min, table of (theta, phi, S2))."""
    def S2_of(x):
        c = bloch(x[0], x[1]); f = superposition(log_psi_1, log_psi_2, c)
        S = sampler(f); half = len(S) // 2
        return renyi2(f, S[:half], S[half:2 * half], mask)[0]

    table = []; best = None
    for th in np.linspace(0, np.pi, grid):
        for ph in np.linspace(0, 2 * np.pi, grid, endpoint=False):
            val = S2_of((th, ph)); table.append((th, ph, val))
            if best is None or val < best[0]:
                best = (val, th, ph)
    x = np.array([best[1], best[2]])
    if refine:
        res = minimize(S2_of, x, method="Nelder-Mead", options=dict(xatol=0.05, fatol=0.01, maxfev=60))
        x = res.x; best = (res.fun, x[0], x[1])
    return bloch(x[0], x[1]), best[0], np.array(table)


def s_matrix(mes_x: Sequence[np.ndarray], mes_y: Sequence[np.ndarray], overlap_12) -> np.ndarray:
    """S_ab = <Xi^x_a | Xi^y_b> from coefficient vectors in the (psi_1, psi_2) basis with Gram matrix [[1, o], [o*, 1]]."""
    G = np.array([[1.0, overlap_12], [np.conj(overlap_12), 1.0]])
    S = np.zeros((2, 2), dtype=complex)
    for a, ca in enumerate(mes_x):
        for b, cb in enumerate(mes_y):
            S[a, b] = np.conj(ca) @ G @ cb
    return S
