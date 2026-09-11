"""Compiled connection rules: for a configuration s, every s' with <s'|O|s> != 0 and the amplitude.

Compilation
-----------
Each term of the operator is reduced to one factor per site (:func:`Operator.canonical`): a diagonal
factor contributes a value that depends on s_i, a raising factor is nonzero only for s_i = -1 and
flips it, a lowering factor only for s_i = +1. Terms are grouped by their *flip set* F (the sites
whose spin changes). For an input s the connected configuration of group g is s with F_g flipped
and its amplitude is the sum over the group's terms of coeff times the product of the per-site
factors -- each a lookup ``table[t, k, (s_i + 1) / 2]``. Every array has a fixed shape (terms padded
with identity factors, groups padded with zero-coefficient terms), so the rule is jit-friendly, and
the number of connected configurations per input is the number of off-diagonal groups.

Matrix-element convention
-------------------------
``amplitudes(s)`` returns <s'|O|s> for the listed s'. The local estimator of an operator O,
O_loc(s) = sum_{s'} <s|O|s'> psi(s')/psi(s), needs <s|O|s'> = conj(<s'|O^dagger|s>), so
:func:`make_local_estimator` compiles O^dagger and conjugates. For a Hermitian Hamiltonian the two
are the same; the general form is used so that non-Hermitian operators (S^+_i S^-_j, ...) are also
estimated correctly.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .operator import Operator


@dataclass
class ConnectionRule:
    N: int
    n_groups: int                    # off-diagonal groups (= connected configurations per input)
    t_sites: np.ndarray              # (n_terms, F_max) int32
    t_table: np.ndarray              # (n_terms, F_max, 2) complex: factor value for s_i = -1 (idx 0) / +1 (idx 1)
    t_coeff: np.ndarray              # (n_terms,) complex
    t_group: np.ndarray              # (n_terms,) int32; group index, n_groups == diagonal
    g_sign: np.ndarray               # (n_groups, N) int8; -1 at the sites flipped by the group
    g_flips: List[Tuple[int, ...]]   # flip set of every group
    n_terms: int = 0
    max_flips: int = 0

    def __post_init__(self):
        self.n_terms = int(self.t_sites.shape[0])
        self.max_flips = max((len(f) for f in self.g_flips), default=0)

    # ------------------------------------------------------------------ numpy path
    def term_values(self, s: np.ndarray) -> np.ndarray:
        """coeff_t prod_k table[t, k, s_{site}] for s (..., N) -> (..., n_terms)."""
        s = np.asarray(s)
        idx = ((s[..., self.t_sites] + 1) // 2).astype(np.int64)            # (..., T, F)
        vals = np.take_along_axis(np.broadcast_to(self.t_table, idx.shape + (2,)), idx[..., None], axis=-1)[..., 0]
        return self.t_coeff * np.prod(vals, axis=-1)

    def amplitudes(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(<s'_g|O|s> for every off-diagonal group (..., n_groups), diagonal <s|O|s> (...,))."""
        v = self.term_values(s)
        lead = v.shape[:-1]
        out = np.zeros(lead + (self.n_groups + 1,), dtype=complex)
        np.add.at(out, (..., self.t_group), v) if v.ndim == 1 else _segment_sum_np(v, self.t_group, self.n_groups + 1, out)
        return out[..., : self.n_groups], out[..., self.n_groups]

    def flips(self, s: np.ndarray) -> np.ndarray:
        """All connected configurations s' (..., n_groups, N)."""
        s = np.asarray(s)
        return (s[..., None, :] * self.g_sign[(None,) * (s.ndim - 1)]).astype(s.dtype)

    def connections(self, s: np.ndarray):
        """(s' (..., n_groups, N), amplitudes (..., n_groups), diagonal (...,)) with zero rows kept."""
        amp, diag = self.amplitudes(s)
        return self.flips(s), amp, diag

    # ------------------------------------------------------------------ jax path
    def jax_arrays(self) -> Dict[str, object]:
        import jax.numpy as jnp
        return dict(t_sites=jnp.asarray(self.t_sites), t_table=jnp.asarray(self.t_table),
                    t_coeff=jnp.asarray(self.t_coeff), t_group=jnp.asarray(self.t_group),
                    g_sign=jnp.asarray(self.g_sign))

    def jax_amplitudes(self) -> Callable:
        """A function s (N,) -> (amp (n_groups,) complex, diag complex) usable under jit/vmap."""
        import jax
        import jax.numpy as jnp
        A = self.jax_arrays()
        t_sites, t_table, t_coeff, t_group = A["t_sites"], A["t_table"], A["t_coeff"], A["t_group"]
        G = self.n_groups

        def amps(s):
            idx = ((s[t_sites] + 1) // 2).astype(jnp.int32)                     # (T, F)
            vals = jnp.take_along_axis(t_table, idx[..., None], axis=-1)[..., 0]  # (T, F)
            v = t_coeff * jnp.prod(vals, axis=-1)                                 # (T,)
            out = jax.ops.segment_sum(v, t_group, num_segments=G + 1)
            return out[:G], out[G]
        return amps

    def jax_flips(self) -> Callable:
        import jax.numpy as jnp
        g_sign = jnp.asarray(self.g_sign)

        def flips(s):
            return (s[None, :] * g_sign).astype(s.dtype)
        return flips

    def __repr__(self) -> str:
        return f"ConnectionRule(N={self.N}, terms={self.n_terms}, groups={self.n_groups}, max_flips={self.max_flips})"


def _segment_sum_np(v, seg, n, out):
    for g in range(n):
        m = seg == g
        if m.any():
            out[..., g] = v[..., m].sum(axis=-1)
    return out


def compile_operator(op: Operator, tol: float = 1e-14) -> ConnectionRule:
    canon = op.canonical(tol)
    groups: Dict[Tuple[int, ...], int] = {}
    g_flips: List[Tuple[int, ...]] = []
    rows = []
    for key, coeff in canon.items():
        sites = []; tables = []; flip = []
        for site, tag in key:
            kind = tag[0]; payload = tag[2:]
            if kind == "d":
                a_up, a_dn = (complex(x) for x in payload.split(","))
                tables.append([a_dn, a_up])              # index 0 <-> s_i = -1 (down), 1 <-> +1 (up)
            elif kind == "p":                            # raise: input must be down
                tables.append([complex(payload), 0.0]); flip.append(site)
            else:                                        # lower: input must be up
                tables.append([0.0, complex(payload)]); flip.append(site)
            sites.append(site)
        flip = tuple(sorted(flip))
        if flip:
            if flip not in groups:
                groups[flip] = len(g_flips); g_flips.append(flip)
            g = groups[flip]
        else:
            g = -1
        rows.append((sites, tables, coeff, g))
    n_groups = len(g_flips)
    F_max = max((len(r[0]) for r in rows), default=1)
    n_terms = max(len(rows), 1)
    t_sites = np.zeros((n_terms, F_max), dtype=np.int32)
    t_table = np.ones((n_terms, F_max, 2), dtype=complex)
    t_coeff = np.zeros(n_terms, dtype=complex)
    t_group = np.full(n_terms, n_groups, dtype=np.int32)
    for a, (sites, tables, coeff, g) in enumerate(rows):
        t_sites[a, : len(sites)] = sites
        t_table[a, : len(sites)] = tables
        t_coeff[a] = coeff
        t_group[a] = n_groups if g < 0 else g
    g_sign = np.ones((n_groups, op.N), dtype=np.int8)
    for g, flip in enumerate(g_flips):
        g_sign[g, list(flip)] = -1
    return ConnectionRule(op.N, n_groups, t_sites, t_table, t_coeff, t_group, g_sign, g_flips)


# ---------------------------------------------------------------------- local estimators
def make_local_estimator(op: Operator, logpsi_batched: Callable, chunk: int = 0, target_configs: int = 65536,
                         compact_fraction: float = 1.0, rule: Optional[ConnectionRule] = None) -> Callable:
    """Return ``oloc(params, s (B, N)) -> complex (B,)`` with O_loc(s) = sum_{s'} <s|O|s'> psi(s')/psi(s).

    ``logpsi_batched(params, s)`` must accept any batch size; if it has an attribute ``cost`` (network
    evaluations per configuration, e.g. the symmetry-group order of a projected state) the batch is
    reduced accordingly. ``chunk`` inputs are processed per jitted call (default: ``target_configs``
    network evaluations). Only groups with nonzero
    amplitude are evaluated: they are compacted into a buffer of capacity
    ``compact_fraction * chunk * n_groups``; if more are needed the chunk returns NaN (lower the
    fraction only when you know the typical fraction of allowed flips, e.g. ~0.55 for exchange
    terms in the singlet sector).
    """
    import jax
    import jax.numpy as jnp

    rule = compile_operator(op.dagger()) if rule is None else rule
    G = rule.n_groups
    if G == 0:                                            # purely diagonal operator
        amps = rule.jax_amplitudes()
        amp_v = jax.jit(jax.vmap(amps))

        def oloc_diag(params, s):
            return jnp.conj(amp_v(s)[1])
        return oloc_diag
    cost = int(getattr(logpsi_batched, "cost", 1))          # network evaluations per configuration (symmetry order)
    if chunk <= 0:
        chunk = max(1, int(target_configs / (G * compact_fraction * cost)))
    cap = int(np.ceil(compact_fraction * chunk * G))
    amp_v = jax.vmap(rule.jax_amplitudes())
    flip_v = jax.vmap(rule.jax_flips())

    @jax.jit
    def core(params, s):
        B = s.shape[0]
        lp0 = logpsi_batched(params, s)                                   # (B,)
        amp, diag = amp_v(s)                                              # (B, G), (B,)
        allowed = (amp != 0).reshape(-1)                                  # (B*G,)
        n_allowed = jnp.sum(allowed)
        idx = jnp.nonzero(allowed, size=cap, fill_value=0)[0]
        valid = jnp.arange(cap) < n_allowed
        sp = flip_v(s).reshape(B * G, -1)[idx]                            # (cap, N)
        lp1 = logpsi_batched(params, sp)                                  # (cap,)
        b_idx = idx // G
        ratio = jnp.where(valid, jnp.exp(lp1 - lp0[b_idx]), 0.0)
        contrib = jnp.conj(amp.reshape(-1)[idx]) * ratio                  # <s|O|s'> = conj(<s'|O^dag|s>)
        offdiag = jax.ops.segment_sum(contrib, b_idx, num_segments=B)
        overflow = n_allowed > cap
        return jnp.where(overflow, jnp.nan, jnp.conj(diag) + offdiag)

    def oloc(params, s):
        B = s.shape[0]
        if B <= chunk:
            return core(params, s)
        return jnp.concatenate([core(params, s[a:a + chunk]) for a in range(0, B, chunk)])

    oloc.chunk = chunk
    oloc.capacity = cap
    oloc.rule = rule
    return oloc


def make_local_energy(hamiltonian: Operator, logpsi_batched: Callable, **kw) -> Callable:
    """Local energy of a Hermitian Hamiltonian (alias of :func:`make_local_estimator`)."""
    return make_local_estimator(hamiltonian, logpsi_batched, **kw)
