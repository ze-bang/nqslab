"""Small ansatze: RBM, Jastrow (pair kernel), MLP, and a fixed-function wrapper."""
from __future__ import annotations

from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def _complex_param(module: nn.Module, name: str, shape, scale: float, init=None):
    if init is not None:
        init = np.asarray(init)
        re = module.param(name + "_re", lambda k, sh: jnp.asarray(np.real(init), dtype=jnp.float64), shape)
        im = module.param(name + "_im", lambda k, sh: jnp.asarray(np.imag(init), dtype=jnp.float64), shape)
    else:
        re = module.param(name + "_re", nn.initializers.normal(scale), shape)
        im = module.param(name + "_im", nn.initializers.normal(scale), shape)
    return re + 1j * im


class RBM(nn.Module):
    """Complex restricted Boltzmann machine: log psi = a.s + sum_j log(2 cosh(b_j + (W s)_j)), alpha hidden units per site."""
    alpha: int = 1
    scale: float = 0.01
    visible_bias: bool = True

    @nn.compact
    def __call__(self, s, aux=None):
        B, N = s.shape
        M = int(self.alpha * N)
        sf = s.astype(jnp.float64)
        W = _complex_param(self, "W", (N, M), self.scale)
        b = _complex_param(self, "b", (M,), self.scale)
        theta = sf @ W + b[None, :]
        out = jnp.sum(jnp.log(2 * jnp.cosh(theta)), axis=1)
        if self.visible_bias:
            a = _complex_param(self, "a", (N,), self.scale)
            out = out + sf @ a
        return out


class Jastrow(nn.Module):
    """Pair kernel: log psi = 1/2 sum_{i != j} U_ij s_i s_j + sum_j v_j n_j.

    mode='disp': U_ij = u[disp_index[i, j]] (translation invariant, n_classes numbers);
    mode='full': U is a symmetric N x N table. Both complex; initialisable from tables.
    """
    disp_index: Optional[np.ndarray] = None
    n_classes: int = 0
    mode: str = "disp"
    u_init: Optional[np.ndarray] = None
    v_init: Optional[np.ndarray] = None
    scale: float = 0.05
    one_body: bool = True

    @nn.compact
    def __call__(self, s, aux=None):
        B, N = s.shape
        if self.mode == "disp":
            u = _complex_param(self, "u", (self.n_classes,), self.scale, self.u_init)
            U = u[jnp.asarray(self.disp_index)]
        else:
            u = _complex_param(self, "u", (N, N), self.scale, self.u_init)
            U = 0.5 * (u + u.T)
        U = U.at[jnp.arange(N), jnp.arange(N)].set(0.0)
        sf = s.astype(jnp.float64)
        out = 0.5 * jnp.einsum("bi,bi->b", sf @ U, sf)
        if self.one_body:
            v = _complex_param(self, "v", (N,), 0.0, self.v_init) if self.v_init is not None else \
                _complex_param(self, "v", (N,), 0.0)
            out = out + (0.5 * (1 + sf)) @ v
        return out


class MLP(nn.Module):
    """Fully connected network s -> (log|psi|, arg psi); optionally symmetric under a permutation set."""
    widths: tuple = (64, 64)

    @nn.compact
    def __call__(self, s, aux=None):
        x = s.astype(jnp.float32)
        if aux is not None:
            x = jnp.concatenate([x, jnp.asarray(aux, dtype=jnp.float32).reshape(x.shape[0], -1)], axis=1)
        for w in self.widths:
            x = nn.gelu(nn.Dense(w)(x))
        out = nn.Dense(2)(x)
        return out[:, 0].astype(jnp.float64) + 1j * out[:, 1].astype(jnp.float64)


class Fixed:
    """A fixed (parameter-free) log-amplitude ``f(s, aux)`` usable as a component of a ProductState."""

    def __init__(self, fn: Callable, name: str = "fixed"):
        self.fn = fn
        self.name = name

    def __call__(self, s, aux=None):
        try:
            return self.fn(s, aux)
        except TypeError:
            return self.fn(s)
