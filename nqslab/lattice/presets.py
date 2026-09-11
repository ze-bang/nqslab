"""Lattice presets. Every constructor takes the torus as an integer supercell matrix, a length-d
sequence of diagonal lengths, or a single integer L (an L^d torus), and returns a
:class:`~nqslab.lattice.lattice.Lattice` with "nn" (and, where standard, "nnn") bond classes and
the natural point operations registered.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .lattice import Lattice

S3 = np.sqrt(3.0)


def _rot2(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _refl2(axis_angle: float) -> np.ndarray:
    c, s = np.cos(2 * axis_angle), np.sin(2 * axis_angle)
    return np.array([[c, s], [s, -c]])


def _affine(R: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float)
    return R, center - R @ center


def _cell(supercell, d: int) -> np.ndarray:
    M = np.asarray(supercell, dtype=int)
    if M.ndim == 0:
        M = np.diag([int(M)] * d)
    elif M.ndim == 1:
        M = np.diag(M)
    return M


def chain(L, nnn: bool = False) -> Lattice:
    lat = Lattice([[1.0]], [[0.0]], _cell(L, 1), name="chain",
                  bond_types={"nn": [(0, 0, (1,))], **({"nnn": [(0, 0, (2,))]} if nnn else {})},
                  point_ops={"inversion": (-np.eye(1), np.zeros(1))})
    return lat


def square(supercell) -> Lattice:
    ops = {"C4": _affine(_rot2(np.pi / 2), [0, 0]), "C2": _affine(_rot2(np.pi), [0, 0]),
           "sigma_x": _affine(_refl2(0.0), [0, 0]), "sigma_d": _affine(_refl2(np.pi / 4), [0, 0])}
    return Lattice(np.eye(2), [[0.0, 0.0]], _cell(supercell, 2), name="square",
                   bond_types={"nn": [(0, 0, (1, 0)), (0, 0, (0, 1))],
                               "nnn": [(0, 0, (1, 1)), (0, 0, (-1, 1))]}, point_ops=ops)


TRIANGULAR_TORI: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    12: ((4, -2), (2, 2)), 21: ((5, -1), (1, 4)), 27: ((6, -3), (3, 3)), 36: ((6, 0), (0, 6)),
    48: ((8, -4), (4, 4)), 108: ((12, -6), (6, 6)), 144: ((12, 0), (0, 12)), 192: ((16, -8), (8, 8))}
"""C6-symmetric triangular tori (T1, T2) in units of (a1, a2), by site number."""


def triangular(supercell=None, N: Optional[int] = None) -> Lattice:
    """Triangular lattice, a1 = (1, 0), a2 = (1/2, sqrt3/2); nn classes a1, a2, a3 = a2 - a1 and the
    three nnn classes; C6 and the reflection across a1 about the origin site."""
    if supercell is None:
        if N not in TRIANGULAR_TORI:
            raise KeyError(f"no tabulated C6 torus for N={N}; give a supercell")
        supercell = TRIANGULAR_TORI[N]
    vec = np.array([[1.0, 0.0], [0.5, S3 / 2]])
    ops = {"C6": _affine(_rot2(np.pi / 3), [0, 0]), "C3": _affine(_rot2(2 * np.pi / 3), [0, 0]),
           "C2": _affine(_rot2(np.pi), [0, 0]), "sigma": _affine(_refl2(0.0), [0, 0])}
    return Lattice(vec, [[0.0, 0.0]], _cell(supercell, 2), name="triangular",
                   bond_types={"nn": [(0, 0, (1, 0)), (0, 0, (0, 1)), (0, 0, (-1, 1))],
                               "nnn": [(0, 0, (1, 1)), (0, 0, (-1, 2)), (0, 0, (-2, 1))]}, point_ops=ops)


def honeycomb(supercell) -> Lattice:
    """Honeycomb: a1 = (1, 0), a2 = (1/2, sqrt3/2), A at 0 and B at (a1 + a2)/3 (bond length 1/sqrt3).
    C6 about the hexagon centre 2(a1 + a2)/3."""
    vec = np.array([[1.0, 0.0], [0.5, S3 / 2]])
    basis = np.array([[0.0, 0.0], (vec[0] + vec[1]) / 3])
    ctr = 2 * (vec[0] + vec[1]) / 3
    ops = {"C6": _affine(_rot2(np.pi / 3), ctr), "C3": _affine(_rot2(2 * np.pi / 3), ctr),
           "C2": _affine(_rot2(np.pi), ctr), "sigma": _affine(_refl2(0.0), ctr)}
    return Lattice(vec, basis, _cell(supercell, 2), name="honeycomb",
                   bond_types={"nn": [(0, 1, (0, 0)), (0, 1, (-1, 0)), (0, 1, (0, -1))],
                               "nnn": [(0, 0, (1, 0)), (0, 0, (0, 1)), (0, 0, (-1, 1)),
                                       (1, 1, (1, 0)), (1, 1, (0, 1)), (1, 1, (-1, 1))]}, point_ops=ops)


def kagome(supercell) -> Lattice:
    """Kagome: a1 = (2, 0), a2 = (1, sqrt3), basis 0, a1/2, a2/2 (bond length 1); C6 about the
    hexagon centre (a1 + a2)/2."""
    vec = np.array([[2.0, 0.0], [1.0, S3]])
    basis = np.array([[0.0, 0.0], vec[0] / 2, vec[1] / 2])
    ctr = (vec[0] + vec[1]) / 2
    ops = {"C6": _affine(_rot2(np.pi / 3), ctr), "C3": _affine(_rot2(2 * np.pi / 3), ctr),
           "C2": _affine(_rot2(np.pi), ctr), "sigma": _affine(_refl2(0.0), ctr)}
    lat = Lattice(vec, basis, _cell(supercell, 2), name="kagome", point_ops=ops)
    lat.add_neighbor_shells(("nn", "nnn"))
    return lat


def cubic(supercell) -> Lattice:
    ops = {"C4z": (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float), np.zeros(3)),
           "C4x": (np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), np.zeros(3)),
           "inversion": (-np.eye(3), np.zeros(3))}
    return Lattice(np.eye(3), [[0.0, 0.0, 0.0]], _cell(supercell, 3), name="cubic",
                   bond_types={"nn": [(0, 0, (1, 0, 0)), (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1))],
                               "nnn": [(0, 0, (1, 1, 0)), (0, 0, (-1, 1, 0)), (0, 0, (1, 0, 1)),
                                       (0, 0, (-1, 0, 1)), (0, 0, (0, 1, 1)), (0, 0, (0, -1, 1))]}, point_ops=ops)


PYROCHLORE_CUBIC_CELL = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
"""Supercell matrix of the conventional cubic cell (4 fcc cells, 16 sites) in the fcc basis."""


def pyrochlore(supercell=None, cubic_cells: Optional[Sequence[int]] = None) -> Lattice:
    """Pyrochlore: fcc a1 = (0, 1, 1)/2, a2 = (1, 0, 1)/2, a3 = (1, 1, 0)/2 (cubic edge 1), basis
    0, a1/2, a2/2, a3/2. ``cubic_cells=(l1, l2, l3)`` builds an l1 x l2 x l3 block of conventional
    cubic cells (16 sites each). Inversion about a site is registered."""
    vec = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    basis = np.array([[0, 0, 0], vec[0] / 2, vec[1] / 2, vec[2] / 2])
    if supercell is None:
        l = np.asarray(cubic_cells if cubic_cells is not None else (1, 1, 1), dtype=int)
        supercell = np.diag(l) @ PYROCHLORE_CUBIC_CELL
    ops = {"inversion": (-np.eye(3), np.zeros(3))}
    lat = Lattice(vec, basis, _cell(supercell, 3), name="pyrochlore", point_ops=ops)
    lat.add_neighbor_shells(("nn",))
    return lat


PRESETS = {"chain": chain, "square": square, "triangular": triangular, "honeycomb": honeycomb,
           "kagome": kagome, "cubic": cubic, "pyrochlore": pyrochlore}


def build(name: str, *args, **kw) -> Lattice:
    return PRESETS[name](*args, **kw)
