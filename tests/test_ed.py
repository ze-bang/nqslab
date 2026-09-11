import numpy as np

from nqslab.lattice import presets
from nqslab.operator import heisenberg, field, total_spin_squared, chirality
from nqslab.ed import diagonalize, expectation_exact, build_sparse, sz_basis, full_basis


def test_square_4x4_heisenberg():
    e, v, basis, codes = diagonalize(heisenberg(presets.square(4)), k=1)
    assert np.isclose(e[0], -11.2284832, atol=1e-6)
    assert abs(expectation_exact(total_spin_squared(16), v[:, 0], basis, codes)) < 1e-8


def test_chain_16():
    e, *_ = diagonalize(heisenberg(presets.chain(16)), k=1)
    assert np.isclose(e[0], -7.1422964, atol=1e-6)


def test_full_space_matches_sector_sum():
    lat = presets.chain(6)
    H = heisenberg(lat)
    e_full, *_ = diagonalize(H, k=64, full=True, dense_below=10 ** 6)
    sectors = np.concatenate([diagonalize(H, n_up=n, k=100, dense_below=10 ** 6)[0] for n in range(7)])
    assert np.allclose(np.sort(e_full), np.sort(sectors))


def test_transverse_field_breaks_sz():
    lat = presets.chain(6)
    H = heisenberg(lat) + field(6, 0.7, "x")
    basis, codes = full_basis(6)
    M = build_sparse(H, basis, codes).toarray()
    assert np.allclose(M, M.conj().T)
    e = np.linalg.eigvalsh(M)
    e_sec = diagonalize(H, n_up=3, k=1)[0][0]   # sector build drops the field's targets: not the true ground state
    assert e[0] < e_sec - 1e-6


def test_chiral_triangular_12_is_chiral():
    lat = presets.triangular(N=12)
    H = heisenberg(lat) + heisenberg(lat, 0.1, "nnn") + chirality(lat, -0.3)
    e, v, basis, codes = diagonalize(H, k=2)
    Hm = heisenberg(lat) + heisenberg(lat, 0.1, "nnn") + chirality(lat, +0.3)
    em, *_ = diagonalize(Hm, k=2)
    assert np.allclose(e, em)                      # time reversal maps Jchi -> -Jchi
    chi = chirality(lat, 1.0)
    assert abs(expectation_exact(chi, v[:, 0], basis, codes).real) > 0.5
