"""Spin-1/2 operators as sums of products of single-site operators.

An :class:`Operator` on N sites is a list of terms ``coeff * o_1(i_1) o_2(i_2) ...`` where each o is
one of ``sp`` (S^+), ``sm`` (S^-), ``sz``, ``sx``, ``sy``, ``id``; the leftmost factor acts last.
Operators support +, -, scalar and operator products, the Hermitian conjugate and a dense matrix
for tests. :func:`compile_operator` turns one into a fixed-shape connection rule (see
:mod:`nqslab.operator.connection`).

Model builders (:func:`heisenberg`, :func:`xxz`, :func:`chirality`, :func:`field`, ...) take a
lattice and bond classes; a twist ``theta`` (length d) multiplies ``S+_j S-_k`` on a bond with
winding w by exp(i theta . w).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# single-site matrices in the (up, down) basis
_M = {
    "id": np.eye(2, dtype=complex),
    "sz": np.array([[0.5, 0], [0, -0.5]], dtype=complex),
    "sp": np.array([[0, 1], [0, 0]], dtype=complex),
    "sm": np.array([[0, 0], [1, 0]], dtype=complex),
    "sx": np.array([[0, 0.5], [0.5, 0]], dtype=complex),
    "sy": np.array([[0, -0.5j], [0.5j, 0]], dtype=complex),
}
_EXPAND = {"sx": ((0.5, "sp"), (0.5, "sm")), "sy": ((-0.5j, "sp"), (0.5j, "sm"))}
_DAGGER = {"sp": "sm", "sm": "sp", "sz": "sz", "sx": "sx", "sy": "sy", "id": "id"}
OPS = tuple(_M)


@dataclass(frozen=True)
class Term:
    ops: Tuple[str, ...]
    sites: Tuple[int, ...]
    coeff: complex

    def __post_init__(self):
        assert len(self.ops) == len(self.sites)
        for o in self.ops:
            if o not in _M:
                raise ValueError(f"unknown operator {o!r}; use one of {OPS}")


class Operator:
    def __init__(self, N: int, terms: Optional[Iterable] = None):
        self.N = int(N)
        self.terms: List[Term] = []
        for t in terms or []:
            self.add(*t) if not isinstance(t, Term) else self.terms.append(t)

    # ------------------------------------------------------------------ construction
    def add(self, ops: Sequence[str], sites: Sequence[int], coeff: complex = 1.0) -> "Operator":
        ops = tuple(ops); sites = tuple(int(s) for s in sites)
        if any(s < 0 or s >= self.N for s in sites):
            raise ValueError(f"site out of range in {sites} (N={self.N})")
        if coeff != 0:
            self.terms.append(Term(ops, sites, complex(coeff)))
        return self

    @staticmethod
    def identity(N: int, coeff: complex = 1.0) -> "Operator":
        return Operator(N, [Term((), (), complex(coeff))])

    def copy(self) -> "Operator":
        return Operator(self.N, list(self.terms))

    def __len__(self) -> int:
        return len(self.terms)

    # ------------------------------------------------------------------ algebra
    def _check(self, other: "Operator"):
        if not isinstance(other, Operator) or other.N != self.N:
            raise ValueError("operators must act on the same number of sites")

    def __add__(self, other):
        if np.isscalar(other):
            return self + Operator.identity(self.N, other)
        self._check(other)
        return Operator(self.N, self.terms + other.terms)

    __radd__ = __add__

    def __neg__(self):
        return Operator(self.N, [Term(t.ops, t.sites, -t.coeff) for t in self.terms])

    def __sub__(self, other):
        return self + (-other if isinstance(other, Operator) else -other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if np.isscalar(other):
            return Operator(self.N, [Term(t.ops, t.sites, t.coeff * complex(other)) for t in self.terms])
        self._check(other)
        return Operator(self.N, [Term(a.ops + b.ops, a.sites + b.sites, a.coeff * b.coeff)
                                 for a in self.terms for b in other.terms])

    def __rmul__(self, other):
        if np.isscalar(other):
            return self * other
        return NotImplemented

    def __matmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1.0 / complex(other))

    def dagger(self) -> "Operator":
        """Hermitian conjugate: reversed factor order, each factor conjugated, coefficient conjugated."""
        return Operator(self.N, [Term(tuple(_DAGGER[o] for o in t.ops[::-1]), t.sites[::-1], np.conj(t.coeff))
                                 for t in self.terms])

    # ------------------------------------------------------------------ canonical form
    def canonical(self, tol: float = 1e-14) -> Dict[Tuple[Tuple[int, str], ...], complex]:
        """Merge terms into a dictionary keyed by the ordered tuple of (site, single-entry factor).

        Every factor is expanded into sp/sm/sz, the product on each site is reduced to one 2x2
        matrix, and each such matrix is either diagonal (``'d:<a_up>,<a_dn>'``), raising, lowering
        or zero. Two operators are equal iff their canonical dictionaries agree.
        """
        out: Dict[Tuple[Tuple[int, str], ...], complex] = defaultdict(complex)
        for t in self.terms:
            for coeff, ops in _expand(t):
                key, c = _canonical_term(ops, t.sites, coeff)
                if key is not None:
                    out[key] += c
        return {k: v for k, v in out.items() if abs(v) > tol}

    def simplify(self, tol: float = 1e-14) -> "Operator":
        """Equivalent operator with merged terms, one product of sp/sm/sz per canonical key.

        A general diagonal factor diag(a_up, a_dn) is split as (a_up + a_dn)/2 id + (a_up - a_dn) sz.
        """
        terms = []
        for key, c in self.canonical(tol).items():
            partial = [(c, [], [])]                              # (coeff, ops, sites)
            for site, tag in key:
                if tag.startswith("d:"):
                    a_up, a_dn = (complex(x) for x in tag[2:].split(","))
                    s0 = 0.5 * (a_up + a_dn); s1 = a_up - a_dn
                    nxt = []
                    for cc, ops, sites in partial:
                        if abs(s0) > tol:
                            nxt.append((cc * s0, list(ops), list(sites)))
                        if abs(s1) > tol:
                            nxt.append((cc * s1, ops + ["sz"], sites + [site]))
                    partial = nxt
                else:
                    op = "sp" if tag.startswith("p:") else "sm"
                    a = complex(tag[2:])
                    partial = [(cc * a, ops + [op], sites + [site]) for cc, ops, sites in partial]
            for cc, ops, sites in partial:
                if abs(cc) > tol:
                    terms.append(Term(tuple(ops), tuple(sites), cc))
        return Operator(self.N, terms)

    def is_hermitian(self, tol: float = 1e-10) -> bool:
        a = self.canonical(tol); b = self.dagger().canonical(tol)
        if set(a) != set(b):
            return False
        return all(abs(a[k] - b[k]) < tol for k in a)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Operator) or other.N != self.N:
            return False
        a = self.canonical(); b = other.canonical()
        return set(a) == set(b) and all(abs(a[k] - b[k]) < 1e-10 for k in a)

    def sites_involved(self) -> np.ndarray:
        return np.array(sorted({s for t in self.terms for s in t.sites}), dtype=int)

    # ------------------------------------------------------------------ dense matrix
    def to_dense(self, basis: Optional[np.ndarray] = None) -> np.ndarray:
        """Dense matrix in the full 2^N basis (bit i = site i up) or in a given basis of configurations."""
        from .connection import compile_operator
        rule = compile_operator(self)
        if basis is None:
            if self.N > 14:
                raise ValueError("full dense matrix only for N <= 14; pass a basis")
            codes = np.arange(2 ** self.N)
            basis = ((codes[:, None] >> np.arange(self.N)[None, :]) & 1) * 2 - 1
        basis = np.asarray(basis, dtype=np.int8)
        codes = (basis == 1).astype(np.int64) @ (1 << np.arange(self.N, dtype=np.int64))
        order = np.argsort(codes)
        sp, amp, diag = rule.connections(basis)
        H = np.zeros((len(basis), len(basis)), dtype=complex)
        H[np.arange(len(basis)), np.arange(len(basis))] += diag
        cp = (sp == 1).astype(np.int64) @ (1 << np.arange(self.N, dtype=np.int64))    # (B, G)
        pos = np.searchsorted(codes[order], cp)
        pos = np.clip(pos, 0, len(basis) - 1)
        rows = order[pos]
        ok = (codes[rows] == cp) & (amp != 0)
        cols = np.broadcast_to(np.arange(len(basis))[:, None], amp.shape)
        np.add.at(H, (rows[ok], cols[ok]), amp[ok])
        return H

    def __repr__(self) -> str:
        head = ", ".join(f"{t.coeff:.3g}*" + "".join(f"{o}({s})" for o, s in zip(t.ops, t.sites)) for t in self.terms[:4])
        more = f", ... ({len(self.terms)} terms)" if len(self.terms) > 4 else ""
        return f"Operator(N={self.N}: {head}{more})"


def _expand(t: Term):
    """Expand sx/sy factors into sp/sm; yields (coeff, ops)."""
    outs = [(t.coeff, [])]
    for o in t.ops:
        if o in _EXPAND:
            outs = [(c * c2, ops + [o2]) for c, ops in outs for c2, o2 in _EXPAND[o]]
        else:
            outs = [(c, ops + [o]) for c, ops in outs]
    for c, ops in outs:
        yield c, tuple(ops)


def _fmt(x: complex) -> str:
    return f"{x.real:.12g}{x.imag:+.12g}j"


def _canonical_term(ops: Tuple[str, ...], sites: Tuple[int, ...], coeff: complex):
    """Reduce a product of sp/sm/sz/id factors to (key, coeff). key = sorted tuple of (site, tag)."""
    per_site: Dict[int, np.ndarray] = {}
    order: List[int] = []
    for o, s in zip(ops, sites):
        if o == "id":
            continue
        if s not in per_site:
            per_site[s] = np.eye(2, dtype=complex); order.append(s)
        per_site[s] = per_site[s] @ _M[o]          # leftmost factor is applied last: M = M_left @ ... @ M_right
    key = []
    for s in sorted(per_site):
        m = per_site[s]
        offd = abs(m[0, 1]) + abs(m[1, 0]); diag = abs(m[0, 0]) + abs(m[1, 1])
        if offd < 1e-15 and diag < 1e-15:
            return None, 0.0
        if offd < 1e-15:
            if abs(m[0, 0] - 1) < 1e-15 and abs(m[1, 1] - 1) < 1e-15:
                continue                             # identity factor
            key.append((s, f"d:{_fmt(m[0, 0])},{_fmt(m[1, 1])}"))
        else:
            assert diag < 1e-15, "a product of single-entry matrices cannot mix diagonal and off-diagonal"
            if abs(m[0, 1]) > 1e-15 and abs(m[1, 0]) > 1e-15:
                raise AssertionError("unexpected two-entry off-diagonal factor")
            if abs(m[0, 1]) > 1e-15:
                key.append((s, f"p:{_fmt(m[0, 1])}"))   # raises: nonzero only for input down
            else:
                key.append((s, f"m:{_fmt(m[1, 0])}"))   # lowers: nonzero only for input up
    return tuple(key), complex(coeff)


# ---------------------------------------------------------------------- builders
def spin_ops(N: int, i: int) -> Dict[str, Operator]:
    """Single-site operators as Operators: {'sp','sm','sz','sx','sy'}."""
    return {o: Operator(N, [Term((o,), (i,), 1.0)]) for o in ("sp", "sm", "sz", "sx", "sy")}


def local(N: int, op: str, i: int, coeff: complex = 1.0) -> Operator:
    return Operator(N, [Term((op,), (i,), coeff)])


def _bond_phase(bonds, theta) -> np.ndarray:
    if theta is None:
        return np.ones(len(bonds), dtype=complex)
    th = np.asarray(theta, dtype=float).reshape(-1)
    return np.exp(1j * (bonds.winding @ th))


def exchange(lattice, J: float, bond_class: str = "nn", theta=None, N: Optional[int] = None) -> Operator:
    """J sum_bonds (S+_j S-_k e^{i theta.w} + h.c.) / 2 -- the transverse (XY) part of a Heisenberg bond."""
    b = lattice.bonds[bond_class]
    N = lattice.N if N is None else N
    H = Operator(N)
    ph = _bond_phase(b, theta)
    for a in range(len(b)):
        j, k = int(b.j[a]), int(b.k[a])
        H.add(("sp", "sm"), (j, k), 0.5 * J * ph[a])
        H.add(("sm", "sp"), (j, k), 0.5 * J * np.conj(ph[a]))
    return H


def ising(lattice, J: float, bond_class: str = "nn", N: Optional[int] = None) -> Operator:
    b = lattice.bonds[bond_class]
    N = lattice.N if N is None else N
    H = Operator(N)
    for a in range(len(b)):
        H.add(("sz", "sz"), (int(b.j[a]), int(b.k[a])), J)
    return H


def heisenberg(lattice, J: float = 1.0, bond_class: str = "nn", theta=None) -> Operator:
    """J sum_bonds S_j . S_k with an optional twist."""
    return exchange(lattice, J, bond_class, theta) + ising(lattice, J, bond_class)


def xxz(lattice, Jzz: float, Jpm: float, bond_class: str = "nn", theta=None) -> Operator:
    """Jzz sum Sz Sz - Jpm sum (S+ S- + S- S+)  (the quantum-spin-ice sign convention)."""
    return ising(lattice, Jzz, bond_class) + exchange(lattice, -2.0 * Jpm, bond_class, theta)


def chirality(lattice, Jchi: float, triangles: Optional[np.ndarray] = None, theta=None,
              bond_class: str = "nn") -> Operator:
    """Jchi sum_triangles S_i . (S_j x S_k) over counter-clockwise triangles (i, j, k).

    S_i.(S_j x S_k) = (i/2) sum_cyclic Sz_a (S+_b S-_c - S-_b S+_c). With a twist, S+_b S-_c
    carries the phase of the (b, c) bond; the bond list is looked up in ``bond_class``.
    """
    tri = lattice.triangles(bond_class) if triangles is None else np.asarray(triangles, dtype=int)
    H = Operator(lattice.N)
    phase = {}
    if theta is not None:
        b = lattice.bonds[bond_class]
        ph = _bond_phase(b, theta)
        for a in range(len(b)):
            phase[(int(b.j[a]), int(b.k[a]))] = ph[a]
            phase[(int(b.k[a]), int(b.j[a]))] = np.conj(ph[a])
    for (i, j, k) in tri:
        for (a, b_, c) in ((i, j, k), (j, k, i), (k, i, j)):
            p = phase.get((int(b_), int(c)), 1.0)
            H.add(("sz", "sp", "sm"), (a, b_, c), 0.5j * Jchi * p)
            H.add(("sz", "sm", "sp"), (a, b_, c), -0.5j * Jchi * np.conj(p))
    return H


def field(N: int, h, axis: str = "z", sites: Optional[Sequence[int]] = None) -> Operator:
    """-sum_i h_i S^axis_i (h scalar or per-site array); axis in {'x','y','z'}."""
    sites = range(N) if sites is None else sites
    h = np.broadcast_to(np.asarray(h, dtype=complex), (N,))
    H = Operator(N)
    for i in sites:
        H.add((f"s{axis}",), (i,), -h[i])
    return H


def total(N: int, op: str = "sz", sites: Optional[Sequence[int]] = None) -> Operator:
    """sum_i op_i."""
    sites = range(N) if sites is None else sites
    return Operator(N, [Term((op,), (int(i),), 1.0) for i in sites])


def total_spin_squared(N: int) -> Operator:
    """S_tot^2 = sum_ij S_i . S_j."""
    H = Operator(N)
    for i in range(N):
        for j in range(N):
            H.add(("sz", "sz"), (i, j), 1.0)
            H.add(("sp", "sm"), (i, j), 0.5)
            H.add(("sm", "sp"), (i, j), 0.5)
    return H
