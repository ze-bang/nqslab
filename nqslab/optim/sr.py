"""Natural-gradient optimisers for VMC.

Notation (real parameters theta, complex log psi, Ns samples, P parameters):
    O_k(s) = d log psi(s)/d theta_k,      X = (O - <O>)/sqrt(Ns)         (Ns x P complex)
    Xt = [Re X; Im X]                      (2Ns x P real)
    eps = (E_loc - <E_loc>)/sqrt(Ns),      eps_t = [Re eps; Im eps]     (2Ns real)
    gradient    g = 2 Xt^T eps_t
    SR metric   S = Xt^T Xt  (P x P),      SR step  dtheta = -lr S^{-1} g / 2 ... written as -2 lr (S + shift)^{-1} Xt^T eps_t
    minSR       dtheta = -2 lr Xt^T (Xt Xt^T + shift)^{-1} eps_t         (2Ns x 2Ns solve; exact when P > 2Ns)

Every optimiser exposes ``step(jac, theta, samples, eloc, extra_local=None) -> (dtheta, info)`` where
``jac`` is a :class:`~nqslab.optim.jacobian.JacobianFn`; :class:`MinSR` also keeps the lower-level
``update(O, eloc, ...)``. Extra local terms (penalties) are added to E_loc before centring. Steps
are clipped to ``max_step_norm`` and rejected (zero) when non-finite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def group_mask(params, lr_groups: Optional[Dict[str, float]]) -> Optional[jnp.ndarray]:
    """Per-parameter learning-rate multipliers from substrings of the parameter path (None if trivial)."""
    if not lr_groups:
        return None
    flat_with_paths = jax.tree_util.tree_flatten_with_path(params)[0]
    mults = []
    for path, leaf in flat_with_paths:
        name = "/".join(str(getattr(p, "key", getattr(p, "name", p))) for p in path)
        m = 1.0
        for key, val in lr_groups.items():
            if key in name:
                m = val
        mults.append(jnp.full(int(np.prod(leaf.shape)), m))
    return jnp.concatenate(mults)


def centred(O, eloc, extra_local=None):
    """(Xt (2Ns, P) float32, eps_t (2Ns,) float64, E, E_err) from O (Ns, P) complex and E_loc (Ns,)."""
    Ns = O.shape[0]
    E = jnp.mean(eloc)
    eloc_t = eloc if extra_local is None else eloc + extra_local
    Et = jnp.mean(eloc_t)
    Om = jnp.mean(O, axis=0, keepdims=True)
    Xt = jnp.concatenate([jnp.real(O - Om), jnp.imag(O - Om)], axis=0).astype(jnp.float32) / jnp.sqrt(Ns)
    eps = (eloc_t - Et) / jnp.sqrt(Ns)
    et = jnp.concatenate([jnp.real(eps), jnp.imag(eps)]).astype(jnp.float64)
    e_err = jnp.std(jnp.real(eloc)) / jnp.sqrt(Ns)
    return Xt, et, E, e_err


@dataclass
class _Base:
    lr: float = 0.02
    diag_shift: float = 1e-3
    max_step_norm: float = 1.0
    mask: Optional[jnp.ndarray] = None
    n_rejected: int = 0

    def _finish(self, dtheta, E, e_err, info: Dict) -> Tuple[jnp.ndarray, Dict]:
        if self.mask is not None:
            dtheta = dtheta * self.mask
        norm = jnp.linalg.norm(dtheta)
        dtheta = jnp.where(norm > self.max_step_norm, dtheta * (self.max_step_norm / norm), dtheta)
        if not bool(jnp.all(jnp.isfinite(dtheta))) or not np.isfinite(float(jnp.real(E))):
            self.n_rejected += 1
            dtheta = jnp.zeros_like(dtheta)
        info = dict(E=float(jnp.real(E)), E_err=float(e_err), step_norm=float(jnp.linalg.norm(dtheta)),
                    rejected=self.n_rejected, lr=self.lr, shift=self.diag_shift, **info)
        return dtheta, info


@dataclass
class MinSR(_Base):
    """minSR: the (2Ns x 2Ns) kernel solve, exact natural gradient when P >= 2Ns."""

    def update(self, O: jnp.ndarray, eloc: jnp.ndarray, extra_local=None, mask=None) -> Tuple[jnp.ndarray, float, float]:
        """Return (dtheta (P,), E, E_err) -- the low-level interface."""
        Xt, et, E, e_err = centred(O, eloc, extra_local)
        del O
        T = (Xt @ Xt.T).astype(jnp.float64)
        T = T + self.diag_shift * jnp.eye(T.shape[0])
        v = jnp.linalg.solve(T, et)
        dtheta = -2.0 * self.lr * (Xt.T @ v.astype(jnp.float32)).astype(jnp.float64)
        if mask is not None:
            dtheta = dtheta * mask
        dtheta, info = self._finish(dtheta, E, e_err, {})
        return dtheta, info["E"], info["E_err"]

    def step(self, jac, theta, samples, eloc, extra_local=None):
        O = jac(theta, samples)
        Xt, et, E, e_err = centred(O, eloc, extra_local)
        del O
        T = (Xt @ Xt.T).astype(jnp.float64) + self.diag_shift * jnp.eye(Xt.shape[0])
        v = jnp.linalg.solve(T, et)
        dtheta = -2.0 * self.lr * (Xt.T @ v.astype(jnp.float32)).astype(jnp.float64)
        return self._finish(dtheta, E, e_err, {})


@dataclass
class ChunkedMinSR(_Base):
    """minSR with the Jacobian held in sample chunks on the host (two passes), for P x 2Ns too large for device memory.

    The kernel T = Xt Xt^T is assembled block by block from chunks of samples; the step
    dtheta = -2 lr sum_chunks Xt_c^T v_c needs the chunks once more. Chunks are cached in host
    memory as float32 (``cache=True``) or recomputed (``cache=False``).
    """
    cache: bool = True

    def step(self, jac, theta, samples, eloc, extra_local=None):
        Ns = samples.shape[0]
        eloc_t = eloc if extra_local is None else eloc + extra_local
        E = jnp.mean(eloc); Et = jnp.mean(eloc_t)
        eps = (eloc_t - Et) / jnp.sqrt(Ns)
        et = np.concatenate([np.real(np.asarray(eps)), np.imag(np.asarray(eps))]).astype(np.float64)
        e_err = jnp.std(jnp.real(eloc)) / jnp.sqrt(Ns)
        # pass 1: mean of O and the chunk list
        chunks = []; Osum = None
        for a, Oc in jac.chunks(theta, samples):
            Osum = Oc.sum(axis=0) if Osum is None else Osum + Oc.sum(axis=0)
            chunks.append((a, np.asarray(Oc) if self.cache else None))
        Om = (Osum / Ns)
        P = Om.shape[0]
        def xt_block(a, Oc):
            Oc = jnp.asarray(Oc) if Oc is not None else next(o for b, o in jac.chunks(theta, samples[a:a + jac.chunk]) if b == 0)
            X = (Oc - Om[None, :]) / jnp.sqrt(Ns)
            return jnp.concatenate([jnp.real(X), jnp.imag(X)], axis=0).astype(jnp.float32)   # (2c, P)
        # pass 2: T blocks. Row ordering of et: [Re (all samples); Im (all samples)] -> map chunk rows accordingly
        T = np.zeros((2 * Ns, 2 * Ns), dtype=np.float64)
        blocks = []
        for a, Oc in chunks:
            Xb = xt_block(a, Oc); c = Xb.shape[0] // 2
            rows = np.concatenate([np.arange(a, a + c), Ns + np.arange(a, a + c)])
            blocks.append((rows, Xb))
        for ra, Xa in blocks:
            for rb, Xb in blocks:
                T[np.ix_(ra, rb)] = np.asarray((Xa @ Xb.T).astype(jnp.float64))
        T += self.diag_shift * np.eye(2 * Ns)
        v = np.linalg.solve(T, et)
        dtheta = jnp.zeros(P, dtype=jnp.float64)
        for rows, Xb in blocks:
            dtheta = dtheta + (Xb.T @ jnp.asarray(v[rows], dtype=jnp.float32)).astype(jnp.float64)
        dtheta = -2.0 * self.lr * dtheta
        return self._finish(dtheta, E, e_err, {})


@dataclass
class SR(_Base):
    """Stochastic reconfiguration with the explicit (P x P) metric (``solver='direct'``) or matrix-free CG."""
    solver: str = "cg"
    cg_maxiter: int = 200
    cg_tol: float = 1e-6

    def step(self, jac, theta, samples, eloc, extra_local=None):
        O = jac(theta, samples)
        Xt, et, E, e_err = centred(O, eloc, extra_local)
        del O
        Xt = Xt.astype(jnp.float64)
        g = Xt.T @ et                                                   # = gradient / 2
        if self.solver == "direct":
            S = Xt.T @ Xt + self.diag_shift * jnp.eye(Xt.shape[1])
            x = jnp.linalg.solve(S, g)
        else:
            mv = lambda v: Xt.T @ (Xt @ v) + self.diag_shift * v
            x, _ = jax.scipy.sparse.linalg.cg(mv, g, maxiter=self.cg_maxiter, tol=self.cg_tol)
        dtheta = -2.0 * self.lr * x
        return self._finish(dtheta, E, e_err, {})


@dataclass
class FirstOrder(_Base):
    """A first-order optax optimiser on the energy gradient g = 2 Xt^T eps_t (``lr`` is passed to optax at construction)."""
    optimizer: object = None                       # an optax GradientTransformation (default adam(lr))
    _state: object = field(default=None, repr=False)

    def __post_init__(self):
        import optax
        if self.optimizer is None:
            self.optimizer = optax.adam(self.lr)

    def step(self, jac, theta, samples, eloc, extra_local=None):
        O = jac(theta, samples)
        Xt, et, E, e_err = centred(O, eloc, extra_local)
        del O
        g = 2.0 * (Xt.astype(jnp.float64).T @ et)
        if self._state is None:
            self._state = self.optimizer.init(theta)
        upd, self._state = self.optimizer.update(g, self._state, theta)
        return self._finish(upd, E, e_err, {})
