"""Bravais lattices with a basis on a torus, in one to three dimensions.

Geometry
--------
* primitive vectors ``vectors`` (d, d), row a is the vector a_a in cartesian coordinates;
* ``basis`` (n_b, d): cartesian offsets of the basis sites inside the unit cell;
* the torus is spanned by T_a = sum_b M[a, b] a_b with an integer matrix ``supercell`` M (d, d);
  the number of unit cells is |det M| and N = |det M| n_b;
* a site is (cell n, sublattice mu) with position r = n . vectors + basis[mu];
* every cell is reduced to the fundamental domain: fractional coordinates f = n M^{-1} in [0, 1),
  the winding w = floor(f) counts how many times the straight path crosses the torus boundaries.
  Twisted boundary conditions couple to the windings of bonds.

Bonds
-----
A bond type is (mu, nu, R): from sublattice mu in cell n to sublattice nu in cell n + R. A named
bond class (e.g. "nn") is a list of bond types; each bond is stored once, oriented as its type says,
with its winding. :meth:`Lattice.neighbor_shells` finds bond types by distance for any lattice.

Symmetries
----------
Translations by every cell vector and any point operation r -> R r + t that maps the lattice
onto itself and is compatible with the torus are returned as site permutations p with
p[i] = image of site i.

Displacement classes
--------------------
``disp_index[i, j]`` labels the pair (i, j) by (mu_i, mu_j, reduced cell displacement); pairs with
the same label are related by a lattice translation. Translation-invariant pair kernels and the
factored attention of the transformer are tables over these ``n_classes = n_b * N`` labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

BondType = Tuple[int, int, Tuple[int, ...]]          # (mu, nu, R)


@dataclass(frozen=True)
class Bonds:
    """Arrays describing one bond class: sites j -> k, windings (n, d), type index (n,)."""
    j: np.ndarray
    k: np.ndarray
    winding: np.ndarray
    type: np.ndarray
    types: Tuple[BondType, ...]

    def __len__(self) -> int:
        return len(self.j)

    @property
    def pairs(self) -> np.ndarray:
        return np.stack([self.j, self.k], axis=1)


class Lattice:
    def __init__(self, vectors, basis, supercell, name: str = "lattice",
                 bond_types: Optional[Dict[str, Sequence[BondType]]] = None,
                 point_ops: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None):
        self.name = name
        self.vectors = np.asarray(vectors, dtype=float)
        self.d = self.vectors.shape[0]
        assert self.vectors.shape == (self.d, self.d)
        self.basis = np.asarray(basis, dtype=float).reshape(-1, self.d)
        self.n_basis = self.basis.shape[0]
        M = np.asarray(supercell, dtype=int)
        if M.ndim == 0 or M.shape == (self.d,):
            M = np.diag(np.broadcast_to(M, (self.d,)))
        assert M.shape == (self.d, self.d), "supercell must be an integer d x d matrix (or a length-d diagonal)"
        det = int(round(np.linalg.det(M)))
        if det == 0:
            raise ValueError("torus vectors are linearly dependent")
        if det < 0 and self.d >= 2:                      # keep a right-handed torus basis
            M = M.copy(); M[[0, 1]] = M[[1, 0]]; det = -det
        self.M = M
        self.n_cells = abs(det)
        self.N = self.n_cells * self.n_basis
        self.Minv = np.linalg.inv(self.M.astype(float))
        self.torus_vectors = self.M @ self.vectors           # rows T_a (cartesian)
        self._build_sites()
        self.bond_types: Dict[str, Tuple[BondType, ...]] = {}
        self.bonds: Dict[str, Bonds] = {}
        for cname, types in (bond_types or {}).items():
            self.add_bond_class(cname, types)
        self._point_ops = dict(point_ops or {})

    # ------------------------------------------------------------------ cells and sites
    def frac(self, cell) -> np.ndarray:
        """Fractional coordinates f of cell vectors n (..., d) with respect to the torus: n = f M."""
        return np.asarray(cell, dtype=float) @ self.Minv

    def reduce(self, cell) -> Tuple[np.ndarray, np.ndarray]:
        """Reduced cell in the fundamental domain and the winding w = floor(f)."""
        cell = np.asarray(cell, dtype=int)
        f = self.frac(cell)
        w = np.floor(f + 1e-9).astype(int)
        return cell - w @ self.M, w

    def _build_sites(self) -> None:
        R = int(np.abs(self.M).sum()) + 2
        rng = [range(-R, R + 1)] * self.d
        seen = set(); cells = []
        for n in np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, self.d):
            key = tuple(int(x) for x in self.reduce(n)[0])
            if key not in seen:
                seen.add(key); cells.append(key)
        assert len(cells) == self.n_cells, (len(cells), self.n_cells)
        cells.sort(key=lambda t: t[::-1])                   # last cell coordinate varies slowest
        self.cells = np.array(cells, dtype=int).reshape(self.n_cells, self.d)
        self._cell_index = {tuple(c): a for a, c in enumerate(cells)}
        self.site_cell = np.repeat(self.cells, self.n_basis, axis=0)          # (N, d)
        self.site_sub = np.tile(np.arange(self.n_basis), self.n_cells)          # (N,)
        self.positions = self.site_cell @ self.vectors + self.basis[self.site_sub]   # (N, d) cartesian

    def cell_index(self, cell) -> Tuple[int, np.ndarray]:
        red, w = self.reduce(cell)
        return self._cell_index[tuple(int(x) for x in red)], w

    def index_of(self, cell, sub: int = 0) -> Tuple[int, Tuple[int, ...]]:
        """Site index of (cell, sublattice) for an arbitrary integer cell, and the winding."""
        c, w = self.cell_index(cell)
        return c * self.n_basis + int(sub), tuple(int(x) for x in w)

    def site(self, i: int) -> Tuple[np.ndarray, int]:
        return self.site_cell[i], int(self.site_sub[i])

    # ------------------------------------------------------------------ bonds
    def add_bond_class(self, name: str, types: Sequence[BondType]) -> Bonds:
        types = tuple((int(mu), int(nu), tuple(int(x) for x in R)) for mu, nu, R in types)
        j = []; k = []; w = []; t = []
        for a, cell in enumerate(self.cells):
            for ti, (mu, nu, R) in enumerate(types):
                i0 = a * self.n_basis + mu
                i1, wind = self.index_of(cell + np.asarray(R), nu)
                j.append(i0); k.append(i1); w.append(wind); t.append(ti)
        b = Bonds(np.array(j, dtype=int), np.array(k, dtype=int), np.array(w, dtype=int).reshape(-1, self.d),
                  np.array(t, dtype=int), types)
        self.bond_types[name] = types
        self.bonds[name] = b
        return b

    def bond_displacement(self, name: str) -> np.ndarray:
        """Cartesian displacement r_k - r_j of every bond of a class, along its type (not reduced)."""
        b = self.bonds[name]
        out = np.empty((len(b), self.d))
        for a, (mu, nu, R) in enumerate(b.types):
            sel = b.type == a
            out[sel] = np.asarray(R) @ self.vectors + self.basis[nu] - self.basis[mu]
        return out

    def neighbor_shells(self, n_shells: int = 2, tol: float = 1e-6, max_range: int = 3) -> List[List[BondType]]:
        """Bond types of the first ``n_shells`` distance shells on the infinite lattice.

        Each undirected pair appears once: (mu, nu, R) is kept when mu < nu, or mu == nu and R is
        lexicographically positive.
        """
        rng = [range(-max_range, max_range + 1)] * self.d
        offsets = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, self.d)
        cand = []
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for R in offsets:
                    if mu > nu or (mu == nu and not _lex_positive(R)):
                        continue
                    dr = R @ self.vectors + self.basis[nu] - self.basis[mu]
                    dist = float(np.linalg.norm(dr))
                    if dist > tol:
                        cand.append((dist, (mu, nu, tuple(int(x) for x in R))))
        cand.sort(key=lambda c: (round(c[0] / tol), c[1]))
        shells: List[List[BondType]] = []
        last = None
        for dist, bt in cand:
            if last is None or dist - last > tol:
                shells.append([]); last = dist
                if len(shells) > n_shells:
                    shells.pop(); break
            shells[-1].append(bt)
        return shells

    def add_neighbor_shells(self, names: Sequence[str] = ("nn", "nnn"), **kw) -> None:
        shells = self.neighbor_shells(len(names), **kw)
        for name, types in zip(names, shells):
            self.add_bond_class(name, types)

    # ------------------------------------------------------------------ geometry
    @cached_property
    def torus_frac(self) -> np.ndarray:
        """Fractional coordinates of the site positions with respect to the torus vectors (N, d)."""
        return self.positions @ np.linalg.inv(self.torus_vectors)

    def site_at(self, r: np.ndarray, tol: float = 1e-6) -> int:
        """Index of the site at cartesian position r modulo the torus; -1 if there is none."""
        f = np.asarray(r, dtype=float) @ np.linalg.inv(self.torus_vectors)
        diff = self.torus_frac - f[None, :]
        diff -= np.round(diff)
        dist = np.linalg.norm(diff @ self.torus_vectors, axis=1)
        a = int(np.argmin(dist))
        return a if dist[a] < tol else -1

    @cached_property
    def distance(self) -> np.ndarray:
        """Minimum-image distance matrix (N, N)."""
        rng = [(-1, 0, 1)] * self.d
        shifts = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, self.d) @ self.torus_vectors
        diff = self.positions[None, :, :] - self.positions[:, None, :]             # (N, N, d)
        dd = np.linalg.norm(diff[None] + shifts[:, None, None, :], axis=-1)        # (S, N, N)
        return dd.min(axis=0)

    @cached_property
    def disp_index(self) -> np.ndarray:
        """Class label of every ordered pair (i, j): (mu_i, mu_j, reduced cell_i - cell_j). Shape (N, N)."""
        D = np.empty((self.N, self.N), dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                c, _ = self.cell_index(self.site_cell[i] - self.site_cell[j])
                D[i, j] = (self.site_sub[i] * self.n_basis + self.site_sub[j]) * self.n_cells + c
        return D

    @property
    def n_classes(self) -> int:
        return self.n_basis * self.n_basis * self.n_cells

    def momenta(self) -> np.ndarray:
        """The n_cells allowed momenta (cartesian, (n_cells, d)): k = m . G with G_a . T_b = 2 pi delta_ab."""
        G = 2 * np.pi * np.linalg.inv(self.torus_vectors).T
        return self.cells @ G

    # ------------------------------------------------------------------ symmetries
    @cached_property
    def translations(self) -> np.ndarray:
        """Permutations of all cell translations, (n_cells, N): perm[c, i] = index of site i shifted by cell c."""
        out = np.empty((self.n_cells, self.N), dtype=int)
        for a, t in enumerate(self.cells):
            for i in range(self.N):
                out[a, i], _ = self.index_of(self.site_cell[i] + t, self.site_sub[i])
        return out

    def permutation_from_map(self, fn: Callable[[np.ndarray], np.ndarray], tol: float = 1e-6) -> Optional[np.ndarray]:
        """Site permutation of a cartesian map r -> fn(r), or None if it does not preserve the torus lattice."""
        perm = np.empty(self.N, dtype=int)
        for i in range(self.N):
            a = self.site_at(fn(self.positions[i]), tol)
            if a < 0:
                return None
            perm[i] = a
        if len(set(perm.tolist())) != self.N:
            return None
        return perm

    def point_operation(self, R, shift=None, tol: float = 1e-6) -> Optional[np.ndarray]:
        """Permutation of the affine map r -> R r + shift (R cartesian (d, d)), or None."""
        R = np.asarray(R, dtype=float)
        t = np.zeros(self.d) if shift is None else np.asarray(shift, dtype=float)
        return self.permutation_from_map(lambda r: R @ r + t, tol)

    def rotation_2d(self, angle: float, center=None) -> Optional[np.ndarray]:
        assert self.d == 2
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        ctr = np.zeros(2) if center is None else np.asarray(center, dtype=float)
        return self.point_operation(R, ctr - R @ ctr)

    def reflection_2d(self, axis_angle: float, center=None) -> Optional[np.ndarray]:
        assert self.d == 2
        c, s = np.cos(2 * axis_angle), np.sin(2 * axis_angle)
        R = np.array([[c, s], [s, -c]])
        ctr = np.zeros(2) if center is None else np.asarray(center, dtype=float)
        return self.point_operation(R, ctr - R @ ctr)

    def inversion(self, center=None) -> Optional[np.ndarray]:
        ctr = np.zeros(self.d) if center is None else np.asarray(center, dtype=float)
        return self.point_operation(-np.eye(self.d), 2 * ctr)

    def point_ops(self) -> Dict[str, np.ndarray]:
        """The preset point operations that are compatible with this torus (name -> permutation)."""
        out = {}
        for name, (R, shift) in self._point_ops.items():
            p = self.point_operation(R, shift)
            if p is not None:
                out[name] = p
        return out

    # ------------------------------------------------------------------ plaquettes
    def triangles(self, bond_class: str = "nn", orientation: str = "ccw") -> np.ndarray:
        """All geometric triangles of a bond class, (n_tri, 3), each stored counter-clockwise (2D only).

        A triangle is three bonds whose true displacements sum to zero (so a small torus with
        boundary-crossing bonds is handled correctly).
        """
        assert self.d == 2, "triangles are defined for two-dimensional lattices"
        b = self.bonds[bond_class]
        disp = self.bond_displacement(bond_class)
        adj: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for a in range(len(b)):
            adj.setdefault(int(b.j[a]), []).append((int(b.k[a]), disp[a]))
            adj.setdefault(int(b.k[a]), []).append((int(b.j[a]), -disp[a]))
        found = set(); out = []
        for i in range(self.N):
            for j, d1 in adj.get(i, []):
                for k, d2 in adj.get(j, []):
                    if k == i:
                        continue
                    for l, d3 in adj.get(k, []):
                        if l == i and np.linalg.norm(d1 + d2 + d3) < 1e-8:
                            cross = d1[0] * d2[1] - d1[1] * d2[0]
                            tri = (i, j, k) if cross > 0 else (i, k, j)
                            m = tri.index(min(tri))
                            key = tri[m:] + tri[:m]
                            if key not in found:
                                found.add(key); out.append(key)
        out.sort()
        tri = np.array(out, dtype=int).reshape(-1, 3)
        if orientation == "cw":
            tri = tri[:, ::-1]
        return tri

    def __repr__(self) -> str:
        return (f"Lattice({self.name}, d={self.d}, n_basis={self.n_basis}, cells={self.n_cells}, N={self.N}, "
                f"bonds={ {k: len(v) for k, v in self.bonds.items()} })")


def _lex_positive(R) -> bool:
    for x in R:
        if x > 0:
            return True
        if x < 0:
            return False
    return False
