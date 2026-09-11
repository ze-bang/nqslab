"""Renyi-2 entropies from the replica swap and region constructions on any 2D lattice."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def swap_estimator(log_psi: Callable, s1: np.ndarray, s2: np.ndarray, mask: np.ndarray) -> Tuple[complex, float]:
    """<Swap_A> = E[ psi(s1_A s2_B) psi(s2_A s1_B) / (psi(s1) psi(s2)) ] with mask = region A (bool, N)."""
    s1 = np.asarray(s1); s2 = np.asarray(s2)
    m = np.asarray(mask, dtype=bool)
    sw1 = np.where(m[None, :], s2, s1); sw2 = np.where(m[None, :], s1, s2)
    lp = np.asarray(log_psi(np.concatenate([s1, s2, sw1, sw2], axis=0)))
    n = len(s1)
    val = np.exp(lp[2 * n:3 * n] + lp[3 * n:] - lp[:n] - lp[n:2 * n])
    val = np.where(np.isfinite(val), val, 0.0)
    return complex(np.mean(val)), float(np.std(val) / np.sqrt(n))


def renyi2(log_psi: Callable, s1: np.ndarray, s2: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    sw, err = swap_estimator(log_psi, s1, s2, mask)
    S2 = -np.log(max(sw.real, 1e-300))
    return float(S2), float(err / max(abs(sw.real), 1e-300))


def min_image_vectors(lattice, center: np.ndarray) -> np.ndarray:
    """Minimum-image displacement of every site from a cartesian centre, (N, d)."""
    d = lattice.d
    rng = [(-1, 0, 1)] * d
    shifts = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, d) @ lattice.torus_vectors
    diff = lattice.positions[None, :, :] + shifts[:, None, :] - np.asarray(center, dtype=float)[None, None, :]
    best = np.argmin(np.linalg.norm(diff, axis=-1), axis=0)
    return diff[best, np.arange(lattice.N)]


def kitaev_preskill_regions(lattice, radius: float, center=None, angle0: float = 30.0) -> Dict[str, np.ndarray]:
    """Three 120-degree sectors A, B, C of a disc; sector boundaries at angle0 + 0, 120, 240 degrees."""
    if center is None:
        center = np.mean(lattice.positions, axis=0)
    v = min_image_vectors(lattice, center)
    r = np.linalg.norm(v, axis=1)
    ang = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) - angle0) % 360.0
    inside = r <= radius + 1e-9
    A = inside & (ang < 120); B = inside & (ang >= 120) & (ang < 240); C = inside & (ang >= 240)
    return dict(A=A, B=B, C=C, AB=A | B, BC=B | C, AC=A | C, ABC=A | B | C)


def levin_wen_regions(lattice, r_in: float, r_out: float, center=None) -> Dict[str, np.ndarray]:
    """Annulus split into an upper half A1 and a lower half A2."""
    if center is None:
        center = np.mean(lattice.positions, axis=0)
    v = min_image_vectors(lattice, center)
    r = np.linalg.norm(v, axis=1)
    ann = (r > r_in) & (r <= r_out)
    return dict(A1=ann & (v[:, 1] >= 0), A2=ann & (v[:, 1] < 0), annulus=ann)


def topological_entropy_kp(log_psi: Callable, s1: np.ndarray, s2: np.ndarray, regions: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """gamma = -(S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC) with Renyi-2 entropies."""
    signs = dict(A=1, B=1, C=1, AB=-1, BC=-1, AC=-1, ABC=1)
    total = 0.0; var = 0.0
    for name, sg in signs.items():
        S, err = renyi2(log_psi, s1, s2, regions[name])
        total += sg * S; var += err ** 2
    return float(-total), float(np.sqrt(var))
