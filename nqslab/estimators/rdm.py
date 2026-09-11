"""Reduced density matrices of small regions by histogramming, and the modular commutator."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def reduced_density_matrix(log_psi: Callable, samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """rho_X(x, x') = E_s[ delta(s_X = x) psi(x', s_Xbar)/psi(x, s_Xbar) ]^*, region X of size m <= ~14."""
    mask = np.asarray(mask, dtype=bool)
    m = int(mask.sum()); idx = np.where(mask)[0]
    S = np.asarray(samples); n = len(S)
    pows = 1 << np.arange(m)
    x_codes = ((S[:, idx] == 1).astype(int) @ pows)
    all_x = ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, :]) & 1) * 2 - 1
    rho = np.zeros((2 ** m, 2 ** m), dtype=complex)
    lp0 = np.asarray(log_psi(S))
    for xp in range(2 ** m):
        Sp = S.copy(); Sp[:, idx] = all_x[xp][None, :]
        r = np.exp(np.asarray(log_psi(Sp)) - lp0)
        r = np.where(np.isfinite(r), r, 0.0)
        np.add.at(rho[:, xp], x_codes, np.conj(r))
    rho /= n
    return 0.5 * (rho + rho.conj().T)


def modular_commutator(rho_ABC: np.ndarray, dims: Tuple[int, int, int]) -> float:
    """J = i Tr(rho_ABC [K_AB, K_BC]) with K = -log rho, regions ordered (A, B, C) in the tensor product."""
    dA, dB, dC = dims
    r = rho_ABC.reshape(dA, dB, dC, dA, dB, dC)
    rho_AB = np.einsum("abcxyc->abxy", r).reshape(dA * dB, dA * dB)
    rho_BC = np.einsum("abcayz->bcyz", r).reshape(dB * dC, dB * dC)
    def K(rho):
        w, v = np.linalg.eigh(rho)
        w = np.clip(w, 1e-14, None)
        return -(v * np.log(w)) @ v.conj().T
    KAB = np.kron(K(rho_AB), np.eye(dC))
    KBC = np.kron(np.eye(dA), K(rho_BC))
    return float((1j * np.trace(rho_ABC @ (KAB @ KBC - KBC @ KAB))).real)
