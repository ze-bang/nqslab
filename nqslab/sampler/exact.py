"""Exact sampling from |psi|^2 over an enumerated basis (small N), for tests and validation."""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..ed.basis import sz_basis, full_basis


class ExactSampler:
    def __init__(self, N: int, n_up: Optional[int] = None, batch: int = 512):
        self.N = N
        self.basis, self.codes = (sz_basis(N, n_up) if n_up is not None else full_basis(N))
        self.batch = batch

    def log_psi_all(self, log_psi: Callable, params, aux=None) -> np.ndarray:
        out = []
        for a in range(0, len(self.basis), self.batch):
            s = jnp.asarray(self.basis[a:a + self.batch])
            out.append(np.asarray(log_psi(params, s) if aux is None else log_psi(params, s, aux)))
        return np.concatenate(out)

    def probabilities(self, log_psi: Callable, params, aux=None) -> np.ndarray:
        lp = self.log_psi_all(log_psi, params, aux)
        w = np.exp(2 * (lp.real - lp.real.max()))
        return w / w.sum()

    def sample(self, log_psi: Callable, params, key, n_samples: int, aux=None) -> jnp.ndarray:
        p = self.probabilities(log_psi, params, aux)
        idx = jax.random.choice(key, len(p), (n_samples,), p=jnp.asarray(p))
        return jnp.asarray(self.basis)[idx]
