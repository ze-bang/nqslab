"""The composed variational state

    psi(s; aux) = P_G [ prod_c psi_c(s; aux) ]

``components`` are flax modules (learnable, each with its own parameter subtree) or
:class:`~nqslab.models.simple.Fixed` callables; ``aux_grid`` (K, n_aux) is an optional grid of
auxiliary feature vectors (e.g. twist angles) selected by an integer index ``k``; ``symmetry``
projects the full product onto a one-dimensional irrep (the projection must act on the product,
not on a single factor, for the result to carry the quantum number).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .simple import Fixed
from .symmetry import SymmetryGroup


class ProductState:
    def __init__(self, N: int, components: Sequence, names: Optional[Sequence[str]] = None,
                 aux_grid: Optional[np.ndarray] = None, symmetry: Optional[SymmetryGroup] = None,
                 aux_features: Optional[Callable] = None):
        self.N = int(N)
        self.components = list(components)
        self.names = list(names) if names is not None else [
            getattr(c, "name", None) or f"c{i}" for i, c in enumerate(self.components)]
        assert len(set(self.names)) == len(self.names), "component names must be unique"
        self.aux_grid = None if aux_grid is None else np.asarray(aux_grid, dtype=float).reshape(len(np.asarray(aux_grid)), -1)
        self.K = 1 if self.aux_grid is None else self.aux_grid.shape[0]
        self.aux_features = aux_features                     # optional map raw aux (B, n) -> features (B, m)
        self.symmetry = symmetry or SymmetryGroup.trivial(self.N)
        self._grid = None if self.aux_grid is None else jnp.asarray(self.aux_grid)

    @property
    def learnable(self) -> List[str]:
        return [n for n, c in zip(self.names, self.components) if not isinstance(c, Fixed)]

    # ------------------------------------------------------------------ parameters
    def init(self, key, batch: int = 2) -> Dict:
        s = jnp.ones((batch, self.N), dtype=jnp.int8).at[:, : self.N // 2].set(-1)
        aux = self._features(jnp.zeros(batch, dtype=int))
        params = {}
        for name, c in zip(self.names, self.components):
            if isinstance(c, Fixed):
                continue
            key, k = jax.random.split(key)
            params[name] = c.init(k, s, aux)
        return params

    @staticmethod
    def n_params(params) -> int:
        return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))

    # ------------------------------------------------------------------ evaluation
    def _features(self, kk):
        if self._grid is None:
            return None
        raw = self._grid[kk]
        return raw if self.aux_features is None else self.aux_features(raw)

    def log_psi(self, params: Dict, s: jnp.ndarray, k=0) -> jnp.ndarray:
        """s (B, N) in {-1, +1}; k an int or (B,) index into the aux grid. Complex (B,)."""
        B = s.shape[0]
        kk = jnp.broadcast_to(jnp.asarray(k), (B,))
        G = self.symmetry.order
        sg = self.symmetry.apply_all(s).reshape(G * B, self.N) if G > 1 else s
        kg = jnp.tile(kk, (G,)) if G > 1 else kk
        aux = self._features(kg)
        lp = jnp.zeros(G * B, dtype=jnp.complex128)
        for name, c in zip(self.names, self.components):
            if isinstance(c, Fixed):
                lp = lp + c(sg, aux)
            else:
                lp = lp + c.apply(params[name], sg, aux)
        if G == 1:
            return lp
        return self.symmetry.project(lp.reshape(G, B))

    def log_psi_fn(self, k=0) -> Callable:
        """log_psi(params, s) at fixed aux index k; carries ``cost`` = symmetry order for the local estimators."""
        fn = lambda params, s: self.log_psi(params, s, k)
        fn.cost = self.symmetry.order
        return fn

    def __call__(self, params, s, k=0):
        return self.log_psi(params, s, k)
