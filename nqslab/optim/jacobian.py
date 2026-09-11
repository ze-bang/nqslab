"""Per-sample log-derivatives O_k(s) = d log psi(s) / d theta_k for real parameters theta and complex log psi."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def flatten(params):
    """(flat real vector, unravel function)."""
    return ravel_pytree(params)


def make_logpsi_flat(state, unravel, k=0) -> Callable:
    """log_psi as a function of the flat parameter vector at aux index k."""
    def f(theta_flat, s):
        return state.log_psi(unravel(theta_flat), s, k)
    return f


class JacobianFn:
    """Compiled O (Ns, P) of complex log psi with respect to the flat real parameters, in sample chunks."""

    def __init__(self, logpsi_flat: Callable, chunk: int = 256, dtype=jnp.complex64):
        def single(th, s1):
            f = lambda t: logpsi_flat(t, s1[None, :])[0]
            gr = jax.grad(lambda t: jnp.real(f(t)))(th)
            gi = jax.grad(lambda t: jnp.imag(f(t)))(th)
            return (gr + 1j * gi).astype(dtype)
        self._vs = jax.jit(jax.vmap(single, in_axes=(None, 0)))
        self.chunk = chunk

    def __call__(self, theta_flat, s):
        B = s.shape[0]; c = self.chunk
        pad = (-B) % c
        if pad:
            s = jnp.concatenate([s, jnp.repeat(s[-1:], pad, axis=0)], axis=0)
        out = [self._vs(theta_flat, s[a:a + c]) for a in range(0, s.shape[0], c)]
        return jnp.concatenate(out, axis=0)[:B]

    def chunks(self, theta_flat, s):
        """Yield (start, O_chunk) without concatenating (for the parameter-chunked optimisers)."""
        B = s.shape[0]; c = self.chunk
        for a in range(0, B, c):
            sb = s[a:a + c]
            pad = c - sb.shape[0]
            if pad:
                sb = jnp.concatenate([sb, jnp.repeat(sb[-1:], pad, axis=0)], axis=0)
            yield a, self._vs(theta_flat, sb)[: min(c, B - a)]
