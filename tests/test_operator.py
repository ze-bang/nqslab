import numpy as np
import pytest

from nqslab.lattice import presets
from nqslab.operator import (Operator, spin_ops, heisenberg, xxz, chirality, field, total, total_spin_squared,
                             compile_operator, exchange, ising)
from nqslab.ed import full_basis, sz_basis


def kron_matrix(op: Operator) -> np.ndarray:
    """Independent dense construction by Kronecker products (bit i = site i, basis index = code)."""
    from nqslab.operator.operator import _M
    N = op.N
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    codes = np.arange(2 ** N)
    bits = (codes[:, None] >> np.arange(N)[None, :]) & 1        # 1 = up
    for t in op.terms:
        M = np.eye(2 ** N, dtype=complex) * t.coeff
        for o, s in zip(t.ops, t.sites):
            m = _M[o]                                              # (up, down) ordering
            full = np.zeros((2 ** N, 2 ** N), dtype=complex)
            for a in codes:
                for bval in (0, 1):                                # new bit at site s
                    b = (a & ~(1 << s)) | (bval << s)
                    full[b, a] = m[1 - bval, 1 - bits[a, s]]
            M = M @ full
        H += M
    return H


def test_spin_algebra():
    o = spin_ops(1, 0)
    assert (o["sx"] * o["sy"] - o["sy"] * o["sx"]) == 1j * o["sz"]
    assert (o["sp"] * o["sm"] - o["sm"] * o["sp"]) == 2 * o["sz"]
    assert (o["sx"] * o["sx"] + o["sy"] * o["sy"] + o["sz"] * o["sz"]) == Operator.identity(1, 0.75)


def test_dense_against_kronecker_random():
    rng = np.random.default_rng(0)
    N = 4
    op = Operator(N)
    for _ in range(25):
        k = rng.integers(1, 4)
        ops = tuple(rng.choice(["sp", "sm", "sz", "sx", "sy", "id"], k))
        sites = tuple(int(x) for x in rng.integers(0, N, k))
        op.add(ops, sites, complex(*rng.normal(size=2)))
    A = op.to_dense(); B = kron_matrix(op)
    assert np.allclose(A, B)
    assert np.allclose(op.dagger().to_dense(), A.conj().T)
    assert np.allclose(op.simplify().to_dense(), A)


def test_two_site_heisenberg_spectrum():
    lat = presets.chain(2)
    # the 2-site chain has a double bond (both windings); build by hand
    H = Operator(2).add(("sz", "sz"), (0, 1)).add(("sp", "sm"), (0, 1), 0.5).add(("sm", "sp"), (0, 1), 0.5)
    assert np.allclose(np.sort(np.linalg.eigvalsh(H.to_dense())), [-0.75, 0.25, 0.25, 0.25])


def test_hermiticity_and_twist():
    lat = presets.triangular(N=12)
    H = heisenberg(lat, 1.0, theta=(0.3, -0.7)) + heisenberg(lat, 0.2, "nnn", theta=(0.3, -0.7)) + chirality(lat, 0.4, theta=(0.3, -0.7))
    assert H.is_hermitian()
    assert not (H + Operator(12).add(("sp",), (0,), 1.0)).is_hermitian()


def test_connection_rule_groups_and_amplitudes():
    lat = presets.square(3)
    H = heisenberg(lat) + field(lat.N, 0.3, "x")
    rule = compile_operator(H)
    assert rule.n_groups == len(lat.bonds["nn"]) + lat.N          # pair flips + single flips
    basis, codes = full_basis(lat.N)
    Hd = H.to_dense()
    # amplitudes: <s'|H|s> for a random s equals the dense column
    s = basis[123]
    sp, amp, diag = rule.connections(s)
    col = Hd[:, 123]
    assert np.isclose(col[123], diag)
    cp = (sp == 1).astype(np.int64) @ (1 << np.arange(lat.N))
    for c, a in zip(cp, amp):
        assert np.isclose(col[c], a)
    assert np.isclose(np.sum(np.abs(col)) - abs(diag), np.sum(np.abs(amp)))


def test_jax_amplitudes_match_numpy():
    import jax, jax.numpy as jnp
    lat = presets.honeycomb(2)
    H = heisenberg(lat) + heisenberg(lat, 0.3, "nnn") + field(lat.N, 0.2, "y")
    rule = compile_operator(H)
    basis, _ = sz_basis(lat.N)
    s = basis[:5]
    amp, diag = rule.amplitudes(s)
    amp_j, diag_j = jax.vmap(rule.jax_amplitudes())(jnp.asarray(s))
    assert np.allclose(np.asarray(amp_j), amp) and np.allclose(np.asarray(diag_j), diag)
    assert np.allclose(np.asarray(jax.vmap(rule.jax_flips())(jnp.asarray(s))), rule.flips(s))


def test_total_spin_and_xxz():
    N = 4
    S2 = total_spin_squared(N).to_dense()
    w = np.round(np.linalg.eigvalsh(S2), 8)
    assert set(w.tolist()) == {0.0, 2.0, 6.0}
    lat = presets.chain(4)
    assert xxz(lat, 1.0, 0.5) == (ising(lat, 1.0) + exchange(lat, -1.0))
