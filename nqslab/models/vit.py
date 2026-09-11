"""Factored-attention vision transformer for lattice spins.

Tokens: one per site, x_i = E[(s_i + 1)/2] (+ an embedding of auxiliary features broadcast to every
token). Layer: x <- x + MHA(LN(x)), x <- x + FFN(LN(x)) with *factored* attention
    head_h(X)_i = sum_j alpha_h[class(i, j)] (X W_V)_j,
one learnable number per displacement class (``lattice.disp_index``): a global convolution that is
exactly translation equivariant on any Bravais lattice with a basis. Pooling: z = sum_i x_i
(translation invariant), a small MLP and two heads, log|psi| and arg psi. Parameters are real;
the network runs in float32 and returns complex128 log psi of shape (B,).
"""
from __future__ import annotations

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class FactoredAttention(nn.Module):
    d: int
    n_heads: int
    disp_index: np.ndarray            # (N, N) int, static
    n_classes: int
    kernel_init_scale: float = 0.5

    @nn.compact
    def __call__(self, x):            # x: (B, N, d)
        B, N, d = x.shape
        H = self.n_heads; dh = d // H
        kern = self.param("kernel", nn.initializers.normal(self.kernel_init_scale), (H, self.n_classes))
        A = kern[:, jnp.asarray(self.disp_index)].astype(jnp.float32)          # (H, N, N)
        V = nn.Dense(d, use_bias=False, name="value")(x).reshape(B, N, H, dh)
        out = jnp.einsum("hij,bjhd->bihd", A, V).reshape(B, N, d)
        return nn.Dense(d, name="out")(out)


class Block(nn.Module):
    d: int
    n_heads: int
    disp_index: np.ndarray
    n_classes: int
    ffn_mult: int = 4

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        x = x + FactoredAttention(self.d, self.n_heads, self.disp_index, self.n_classes)(h)
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.ffn_mult * self.d)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d)(h)
        return x + h


class FactoredViT(nn.Module):
    """log psi of shape (B,) for s of shape (B, N) in {-1, +1} and optional aux features (B, n_aux)."""
    disp_index: np.ndarray
    n_classes: int
    d: int = 32
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: int = 4
    d_head_mlp: int = 64
    n_aux: int = 0                    # number of auxiliary features (0: none)
    sublattice: Optional[np.ndarray] = None   # (N,) sublattice index of every site (adds a sublattice embedding)

    @nn.compact
    def __call__(self, s, aux=None):
        B, N = s.shape
        idx = ((s + 1) // 2).astype(jnp.int32)
        x = nn.Embed(2, self.d, name="embed")(idx).astype(jnp.float32)              # (B, N, d)
        if self.sublattice is not None:
            n_sub = int(np.max(self.sublattice)) + 1
            if n_sub > 1:
                x = x + nn.Embed(n_sub, self.d, name="sublattice_embed")(jnp.asarray(self.sublattice))[None].astype(jnp.float32)
        if self.n_aux > 0 and aux is not None:
            t = nn.Dense(self.d, name="aux_embed")(jnp.asarray(aux, dtype=jnp.float32).reshape(B, self.n_aux))
            x = x + t[:, None, :]
        for l in range(self.n_layers):
            x = Block(self.d, self.n_heads, self.disp_index, self.n_classes, self.ffn_mult, name=f"block{l}")(x)
        x = nn.LayerNorm(name="final_norm")(x)
        z = jnp.sum(x, axis=1)                                                        # (B, d)
        rho = nn.gelu(nn.Dense(self.d_head_mlp, name="head_mlp")(z))
        log_abs = nn.Dense(1, name="head_amplitude")(rho)[:, 0]
        phase = nn.Dense(1, name="head_phase")(rho)[:, 0]
        return log_abs.astype(jnp.float64) + 1j * phase.astype(jnp.float64)


def vit_for(lattice, **kw) -> FactoredViT:
    """A FactoredViT wired to a lattice (displacement classes and sublattices)."""
    return FactoredViT(disp_index=lattice.disp_index, n_classes=lattice.n_classes,
                       sublattice=lattice.site_sub if lattice.n_basis > 1 else None, **kw)
