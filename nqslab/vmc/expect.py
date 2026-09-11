"""Expectation values from samples."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from ..operator import Operator, make_local_estimator


def expectation(op: Operator, logpsi_batched: Callable, params, samples, estimator: Optional[Callable] = None,
                **kw) -> Tuple[complex, float]:
    """<O> and its statistical error from samples of |psi|^2. Pass a cached ``estimator`` to skip recompilation."""
    est = make_local_estimator(op, logpsi_batched, **kw) if estimator is None else estimator
    vals = np.asarray(est(params, jnp.asarray(samples)))
    vals = vals[np.isfinite(vals)]
    return complex(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


def sz_correlations(samples) -> np.ndarray:
    """<s_i s_j> / 4 = <Sz_i Sz_j> from samples (N, N)."""
    S = np.asarray(samples, dtype=float)
    return (S.T @ S) / (4.0 * len(S))


def structure_factor(lattice, corr: np.ndarray, momenta: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """S(q) = 1/N sum_ij C_ij exp(i q.(r_i - r_j)) at the allowed momenta (or given q). Returns (q, S(q))."""
    q = lattice.momenta() if momenta is None else np.asarray(momenta)
    r = lattice.positions
    phase = np.exp(1j * (r @ q.T))                                     # (N, nq)
    S = np.einsum("iq,ij,jq->q", np.conj(phase), corr, phase).real / lattice.N
    return q, S
