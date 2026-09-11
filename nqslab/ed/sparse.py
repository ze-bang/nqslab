"""Sparse exact diagonalization from the compiled connection rule (so ED and VMC agree by construction).

Suitable for N <= ~26 in a fixed-S^z sector; larger systems go through :mod:`nqslab.ed.qed_adapter`.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..operator import Operator, ConnectionRule, compile_operator
from .basis import sz_basis, full_basis, codes_of


def build_sparse(op: Union[Operator, ConnectionRule], basis: np.ndarray, codes: np.ndarray, chunk: int = 20000) -> sp.csr_matrix:
    """Matrix <s'|O|s> of an operator in a basis (rows s', columns s); codes must be sorted."""
    rule = compile_operator(op) if isinstance(op, Operator) else op
    N = basis.shape[1]; dim = len(basis)
    pows = (1 << np.arange(N, dtype=np.int64))
    g_xor = ((rule.g_sign == -1).astype(np.int64) @ pows) if rule.n_groups else np.zeros(0, dtype=np.int64)
    rows, cols, vals = [], [], []
    for start in range(0, dim, chunk):
        s = basis[start:start + chunk]
        B = len(s)
        amp, diag = rule.amplitudes(s)                                          # (B, G), (B,)
        if rule.n_groups:
            flipped = codes[start:start + B][:, None] ^ g_xor[None, :]         # (B, G)
            nz = np.nonzero(amp)
            pos = np.searchsorted(codes, flipped[nz])
            pos = np.clip(pos, 0, dim - 1)
            ok = codes[pos] == flipped[nz]                                      # drop targets outside the basis
            rows.append(pos[ok]); cols.append((start + nz[0])[ok]); vals.append(amp[nz][ok])
        rows.append(np.arange(start, start + B)); cols.append(np.arange(start, start + B)); vals.append(diag.astype(complex))
    H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    return H.tocsr()


def ground_states(H: sp.csr_matrix, k: int = 4, tol: float = 1e-10, dense_below: int = 400):
    if H.shape[0] < dense_below:
        e, v = np.linalg.eigh(H.toarray())
        return e[:k], v[:, :k]
    e, v = spla.eigsh(H, k=k, which="SA", tol=tol)
    order = np.argsort(e)
    return e[order], v[:, order]


def diagonalize(op: Operator, n_up: Optional[int] = None, k: int = 4, full: bool = False, **kw):
    """Convenience: (energies, vectors, basis, codes) of an operator in a fixed-S^z sector (default N//2) or the full space."""
    basis, codes = full_basis(op.N) if full else sz_basis(op.N, n_up)
    H = build_sparse(op, basis, codes)
    e, v = ground_states(H, k=k, **kw)
    return e, v, basis, codes


def dense_state_vector(log_psi: Callable[[np.ndarray], np.ndarray], basis: np.ndarray, chunk: int = 5000) -> np.ndarray:
    """Normalised vector of a wavefunction given as log psi on the basis."""
    out = np.empty(len(basis), dtype=complex)
    for start in range(0, len(basis), chunk):
        out[start:start + chunk] = log_psi(basis[start:start + chunk])
    out = np.exp(out - np.max(out.real))
    return out / np.linalg.norm(out)


def expectation_exact(op: Operator, vec: np.ndarray, basis: np.ndarray, codes: np.ndarray) -> complex:
    """<vec|O|vec> for a normalised vector in the basis."""
    return complex(np.vdot(vec, build_sparse(op, basis, codes) @ vec))


def apply_operator(op: Operator, vec: np.ndarray, basis: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return build_sparse(op, basis, codes) @ vec
