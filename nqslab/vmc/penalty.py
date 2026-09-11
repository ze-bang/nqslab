"""Penalty terms added to the local energy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax.numpy as jnp


@dataclass
class OrthogonalityPenalty:
    """lambda |<ref|psi>|^2 / (<ref|ref><psi|psi>) as the local term lambda (psi_ref(s)/psi(s)) B, B = E_ref[psi/psi_ref].

    Used to target excited states / other sectors: minimising E + penalty pushes psi orthogonal to
    a fixed reference state given by ``ref_log_psi`` and samples ``ref_samples`` from |psi_ref|^2.
    """
    ref_log_psi: Callable
    lam: float = 5.0
    ref_samples: Optional[jnp.ndarray] = None

    def local_term(self, model_log_psi: Callable, s: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        lp = model_log_psi(s); lr = self.ref_log_psi(s)
        Bv = jnp.mean(jnp.exp(model_log_psi(self.ref_samples) - self.ref_log_psi(self.ref_samples)))
        loc = self.lam * jnp.exp(lr - lp) * Bv
        fid = jnp.mean(jnp.exp(lr - lp)) * Bv
        return loc, float(jnp.real(fid))
