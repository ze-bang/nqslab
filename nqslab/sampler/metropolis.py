"""Vectorised Metropolis sampling with pluggable move sets.

A move set proposes s' from s for every chain in parallel and reports whether the proposal is
symmetric (all built-in moves are). The sampler runs ``n_chains`` chains under ``jax.lax.scan``,
one proposal per step, ``sweep_size`` steps per sweep, and returns the configurations after every
sweep (thinned).

Moves
-----
* :class:`ExchangeMove`   swap two opposite spins: a random pair from a bond list with probability
                          1 - p_long, any pair otherwise. Conserves total S^z.
* :class:`FlipMove`       flip one random spin. Changes total S^z (needed for transverse fields).
* :class:`MixedMove`      choose among moves with given probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class Move:
    def propose(self, s: jnp.ndarray, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """(s' (C, N), proposal_allowed (C,) bool). Symmetric proposals only."""
        raise NotImplementedError

    @property
    def conserves_sz(self) -> bool:
        return False


@dataclass
class ExchangeMove(Move):
    pairs: np.ndarray                       # (n_pairs, 2) candidate pairs (e.g. lattice.bonds['nn'].pairs)
    p_long: float = 0.0                     # probability of proposing an arbitrary pair of sites

    def __post_init__(self):
        self._pairs = jnp.asarray(np.asarray(self.pairs, dtype=int).reshape(-1, 2))

    @property
    def conserves_sz(self) -> bool:
        return True

    def propose(self, s, key):
        C, N = s.shape
        k1, k2, k3, k4 = jax.random.split(key, 4)
        b = jax.random.randint(k2, (C,), 0, self._pairs.shape[0])
        i_b, j_b = self._pairs[b, 0], self._pairs[b, 1]
        if self.p_long > 0:
            use_long = jax.random.uniform(k1, (C,)) < self.p_long
            i_l = jax.random.randint(k3, (C,), 0, N)
            j_l = (i_l + jax.random.randint(k4, (C,), 1, N)) % N
            i = jnp.where(use_long, i_l, i_b); j = jnp.where(use_long, j_l, j_b)
        else:
            i, j = i_b, j_b
        rows = jnp.arange(C)
        si, sj = s[rows, i], s[rows, j]
        sp = s.at[rows, i].set(sj).at[rows, j].set(si)
        return sp, si != sj


@dataclass
class FlipMove(Move):
    def propose(self, s, key):
        C, N = s.shape
        i = jax.random.randint(key, (C,), 0, N)
        rows = jnp.arange(C)
        return s.at[rows, i].multiply(-1), jnp.ones(C, dtype=bool)


@dataclass
class MixedMove(Move):
    moves: Sequence[Move]
    probs: Sequence[float]

    def __post_init__(self):
        p = np.asarray(self.probs, dtype=float); self._cum = jnp.asarray(np.cumsum(p / p.sum()))

    @property
    def conserves_sz(self) -> bool:
        return all(m.conserves_sz for m in self.moves)

    def propose(self, s, key):
        C = s.shape[0]
        keys = jax.random.split(key, len(self.moves) + 1)
        u = jax.random.uniform(keys[0], (C,))
        which = jnp.sum(u[:, None] >= self._cum[None, :], axis=1)
        sp = s; ok = jnp.zeros(C, dtype=bool)
        for a, m in enumerate(self.moves):
            s_a, ok_a = m.propose(s, keys[a + 1])
            sel = which == a
            sp = jnp.where(sel[:, None], s_a, sp); ok = jnp.where(sel, ok_a, ok)
        return sp, ok


@dataclass
class MetropolisSampler:
    """Metropolis chains for |psi(s)|^2.

    ``log_psi(params, s, aux)`` is evaluated on a batch (n_chains, N); ``aux`` is passed through
    unchanged (an auxiliary index such as a twist point).
    """
    N: int
    move: Move
    n_chains: int = 64
    sweep_size: int = 0                     # proposals per sweep (default N)

    def __post_init__(self):
        if self.sweep_size == 0:
            self.sweep_size = self.N

    def init_state(self, key, n_up: Optional[int] = None) -> jnp.ndarray:
        """Random configurations, with n_up up spins per chain if given (default N // 2 when the move conserves S^z)."""
        N = self.N
        if n_up is None and self.move.conserves_sz:
            n_up = N // 2
        if n_up is None:
            return (2 * jax.random.bernoulli(key, 0.5, (self.n_chains, N)).astype(jnp.int8) - 1)

        def one(k):
            perm = jax.random.permutation(k, N)
            return (-jnp.ones(N, dtype=jnp.int8)).at[perm[:n_up]].set(1)
        return jax.vmap(one)(jax.random.split(key, self.n_chains))

    def make_sampler(self, log_psi: Callable, aux=None) -> Callable:
        """Return jitted ``sample(params, s0, key, n_sweeps, thin, n_burn) -> (samples, s_last, acceptance)``.

        samples has shape ((n_sweeps // thin) * n_chains, N): the chain states after every thin-th sweep.
        """
        N, C = self.N, self.n_chains
        move = self.move; sweep = self.sweep_size
        lp_fn = (lambda params, s: log_psi(params, s)) if aux is None else (lambda params, s: log_psi(params, s, aux))

        def sample(params, s0, key, n_sweeps: int, thin: int = 1, n_burn: int = 0):
            def step(carry, key):
                s, lp, acc = carry
                k1, k2 = jax.random.split(key)
                sp, ok = move.propose(s, k1)
                lpp = lp_fn(params, sp)
                ratio = jnp.exp(2.0 * jnp.real(lpp - lp))
                accept = ok & (jax.random.uniform(k2, (C,)) < ratio) & jnp.isfinite(lpp)
                s = jnp.where(accept[:, None], sp, s)
                lp = jnp.where(accept, lpp, lp)
                return (s, lp, acc + accept.mean()), None

            def sweep_fn(carry, key):
                carry, _ = jax.lax.scan(step, carry, jax.random.split(key, sweep))
                return carry, carry[0]

            carry = (s0, lp_fn(params, s0), 0.0)
            if n_burn > 0:
                kb, key = jax.random.split(key)
                carry, _ = jax.lax.scan(sweep_fn, carry, jax.random.split(kb, n_burn))
                carry = (carry[0], carry[1], 0.0)
            carry, samples = jax.lax.scan(sweep_fn, carry, jax.random.split(key, n_sweeps))
            samples = samples[thin - 1::thin].reshape(-1, N)
            return samples, carry[0], carry[2] / (n_sweeps * sweep)

        return jax.jit(sample, static_argnums=(3, 4, 5))


def exchange_sampler(lattice, n_chains: int = 64, p_long: float = 0.1, bond_class: str = "nn", **kw) -> MetropolisSampler:
    """S^z-conserving exchange sampler over the bonds of a lattice class (plus long-range swaps)."""
    return MetropolisSampler(lattice.N, ExchangeMove(lattice.bonds[bond_class].pairs, p_long), n_chains, **kw)


def flip_sampler(N: int, n_chains: int = 64, **kw) -> MetropolisSampler:
    return MetropolisSampler(N, FlipMove(), n_chains, **kw)
