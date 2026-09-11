import numpy as np
import jax, jax.numpy as jnp

from nqslab.lattice import presets
from nqslab.operator import heisenberg, chirality
from nqslab.ed import diagonalize, sz_basis
from nqslab.estimators import (overlap_ratio, fidelity, renyi2, reduced_density_matrix, kitaev_preskill_regions,
                               plaquette_berry_phase, exact, numpy_exchange_sampler, superposition, bloch)


def _vec_fn(v, codes, N):
    tbl = np.full(2 ** N, -80.0 + 0j); tbl[codes] = np.log(v.astype(complex) + 1e-300)
    pows = 1 << np.arange(N)
    return lambda s: tbl[(np.asarray(s) == 1).astype(int) @ pows]


def test_overlap_fidelity_renyi_rdm_against_exact():
    lat = presets.triangular(N=12); N = lat.N
    H = heisenberg(lat) + heisenberg(lat, 0.1, "nnn") + chirality(lat, -0.3)
    e, v, basis, codes = diagonalize(H, k=2)
    f0 = _vec_fn(v[:, 0], codes, N); f1 = _vec_fn(v[:, 1], codes, N)
    rng = np.random.default_rng(0)
    S0 = numpy_exchange_sampler(f0, lat.bonds["nn"].pairs, N, 64, 60, rng)
    S1 = numpy_exchange_sampler(f1, lat.bonds["nn"].pairs, N, 64, 60, rng)
    o, err = overlap_ratio(f0, f1, S0)
    assert abs(o) < 5 * err + 0.03
    fid, ferr = fidelity(f0, f0, S0[: len(S0) // 2], S0[len(S0) // 2:])
    assert abs(fid - 1) < 1e-6
    mask = np.zeros(N, dtype=bool); mask[:4] = True
    half = len(S0) // 2
    S2, s2err = renyi2(f0, S0[:half], S0[half:], mask)
    assert abs(S2 - exact.renyi2_exact(v[:, 0], codes, N, mask)) < 5 * s2err + 0.05
    rho = reduced_density_matrix(f0, S0, np.arange(N) < 3)
    rho_ex = exact.rdm_exact(v[:, 0], codes, N, np.arange(N) < 3)
    assert np.abs(rho - rho_ex).max() < 0.05
    regs = kitaev_preskill_regions(lat, 1.1)
    assert regs["ABC"].sum() > 0


def test_berry_phase_sampled_matches_exact():
    lat = presets.triangular(N=12); N = lat.N
    d = 0.4
    corners = [(0, 0), (d, 0), (d, d), (0, d)]
    states = []; fns = []; samples = []
    rng = np.random.default_rng(1)
    for th in corners:
        H = heisenberg(lat, theta=th) + heisenberg(lat, 0.1, "nnn", theta=th) + chirality(lat, -0.3, theta=th)
        e, v, basis, codes = diagonalize(H, k=1)
        states.append(v[:, :1]); f = _vec_fn(v[:, 0], codes, N); fns.append([f])
        samples.append([numpy_exchange_sampler(f, lat.bonds["nn"].pairs, N, 64, 40, rng)])
    F_ex = exact.plaquette_berry_phase_exact(states)
    F, _ = plaquette_berry_phase(fns, samples)
    assert abs(np.angle(np.exp(1j * (F - F_ex)))) < 0.1
