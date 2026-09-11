import numpy as np
import jax, jax.numpy as jnp
import pytest

from nqslab.lattice import presets
from nqslab.operator import heisenberg, field, make_local_energy, make_local_estimator, total_spin_squared, Operator
from nqslab.ed import diagonalize, dense_state_vector, expectation_exact, sz_basis, full_basis
from nqslab.models import vit_for, Jastrow, RBM, Fixed, ProductState, SymmetryGroup, translation_group, point_group, marshall_sign
from nqslab.sampler import ExactSampler, exchange_sampler, flip_sampler
from nqslab.optim import MinSR, ChunkedMinSR, SR, FirstOrder, JacobianFn, flatten, make_logpsi_flat, centred
from nqslab.vmc import VMC, VMCConfig, expectation, OrthogonalityPenalty


def exact_state(N, v, codes):
    tbl = np.full(2 ** N, -60.0 + 0j); tbl[codes] = np.log(v.astype(complex) + 1e-300)
    tbl_j = jnp.asarray(tbl); pows = jnp.asarray(1 << np.arange(N))
    return ProductState(N, [Fixed(lambda s, aux=None: tbl_j[((s == 1).astype(jnp.int64) @ pows)])])


def test_local_energy_zero_variance_and_operator_estimator():
    lat = presets.square(3); N = lat.N
    H = heisenberg(lat) + heisenberg(lat, 0.3, "nnn")
    e, v, basis, codes = diagonalize(H, n_up=4, k=1)
    st = exact_state(N, v[:, 0], codes)
    S = ExactSampler(N, 4).sample(st.log_psi, {}, jax.random.PRNGKey(0), 300, aux=0)
    E = np.asarray(make_local_energy(H, st.log_psi_fn(0))({}, S))
    assert np.allclose(E, e[0], atol=1e-8)
    # a non-Hermitian operator: <S+_0 S-_1> from samples vs exact
    O = Operator(N).add(("sp", "sm"), (0, 1), 1.0)
    val, err = expectation(O, st.log_psi_fn(0), {}, ExactSampler(N, 4).sample(st.log_psi, {}, jax.random.PRNGKey(1), 4000, aux=0))
    ref = expectation_exact(O, v[:, 0], basis, codes)
    assert abs(val - ref) < 5 * err + 1e-3


def test_local_energy_with_transverse_field_full_space():
    lat = presets.chain(6); N = lat.N
    H = heisenberg(lat) + field(N, 0.6, "x")
    e, v, basis, codes = diagonalize(H, k=1, full=True)
    st = exact_state(N, v[:, 0], codes)
    S = ExactSampler(N).sample(st.log_psi, {}, jax.random.PRNGKey(0), 300, aux=0)
    E = np.asarray(make_local_energy(H, st.log_psi_fn(0))({}, S))
    assert np.allclose(E, e[0], atol=1e-8)


def test_jacobian_matches_finite_differences():
    lat = presets.chain(6); N = lat.N
    st = ProductState(N, [RBM(alpha=1, scale=0.2), Jastrow(disp_index=lat.disp_index, n_classes=lat.n_classes)])
    p = st.init(jax.random.PRNGKey(0)); theta, unravel = flatten(p)
    f = make_logpsi_flat(st, unravel, 0)
    s = jnp.asarray(sz_basis(N)[0][:3])
    O = np.asarray(JacobianFn(f, chunk=2, dtype=jnp.complex128)(theta, s))
    eps = 1e-6
    for k in [0, 5, len(theta) - 1]:
        dp = theta.at[k].add(eps); dm = theta.at[k].add(-eps)
        fd = (np.asarray(f(dp, s)) - np.asarray(f(dm, s))) / (2 * eps)
        assert np.allclose(O[:, k], fd, atol=1e-5)


def test_minsr_sr_and_chunked_agree():
    lat = presets.chain(6); N = lat.N
    H = heisenberg(lat)
    st = ProductState(N, [RBM(alpha=1, scale=0.2)])
    p = st.init(jax.random.PRNGKey(0)); theta, unravel = flatten(p)
    jac = JacobianFn(make_logpsi_flat(st, unravel, 0), chunk=8, dtype=jnp.complex128)
    S = ExactSampler(N, 3).sample(st.log_psi, p, jax.random.PRNGKey(1), 40, aux=0)
    E = make_local_energy(H, st.log_psi_fn(0))(p, S)
    kw = dict(lr=0.01, diag_shift=1e-4, max_step_norm=1e9)
    d1, _ = MinSR(**kw).step(jac, theta, S, E)
    d2, _ = SR(solver="direct", **kw).step(jac, theta, S, E)
    d3, _ = SR(solver="cg", cg_tol=1e-10, cg_maxiter=2000, **kw).step(jac, theta, S, E)
    d4, _ = ChunkedMinSR(**kw).step(jac, theta, S, E)
    # minSR and SR solve the same regularised system only up to the shift placement; compare at tiny shift
    assert np.allclose(np.asarray(d1), np.asarray(d4), rtol=1e-3, atol=1e-6)
    assert np.allclose(np.asarray(d2), np.asarray(d3), rtol=1e-3, atol=1e-6)
    assert np.allclose(np.asarray(d1), np.asarray(d2), rtol=5e-2, atol=1e-4)


def test_first_order_step_descends():
    lat = presets.chain(6); N = lat.N
    H = heisenberg(lat)
    st = ProductState(N, [RBM(alpha=1, scale=0.2)])
    run = VMC(st, H, ExactSampler(N, 3), FirstOrder(lr=0.02), VMCConfig(steps=30, n_samples=400, log_every=100))
    hist = run.run(verbose=False)
    assert hist[-1]["E"] < hist[0]["E"]


def test_vmc_reaches_ground_state_small_chain():
    lat = presets.chain(8); N = lat.N
    H = heisenberg(lat)
    e0 = diagonalize(H, k=1)[0][0]
    G = translation_group(lat, 0).times(SymmetryGroup.spin_flip(N, 1))
    # A symmetry-projected RBM started from positive amplitudes plateaus at E = -2.99 (sign-learning trap,
    # see docs/design.md); the Marshall sign prior removes it. Without projection the RBM converges either way.
    st = ProductState(N, [RBM(alpha=2, scale=0.05), Jastrow(disp_index=lat.disp_index, n_classes=lat.n_classes),
                          marshall_sign(np.arange(0, N, 2))], symmetry=G)
    cfg = VMCConfig(steps=150, n_samples=512, lr=0.02, lr_final=0.01, diag_shift=1e-3, diag_shift_final=1e-4, log_every=100)
    run = VMC(st, H, ExactSampler(N, N // 2), MinSR(lr=0.02, diag_shift=1e-3, max_step_norm=1.0), cfg)
    hist = run.run(verbose=False)
    assert abs(run.energy - e0) / abs(e0) < 2e-2, [round(h["E"], 3) for h in hist[::10]]


def test_orthogonality_penalty_targets_excited_state():
    lat = presets.chain(6); N = lat.N
    H = heisenberg(lat)
    e, v, basis, codes = diagonalize(H, k=2)
    ref = exact_state(N, v[:, 0], codes)
    ref_samples = ExactSampler(N, 3).sample(ref.log_psi, {}, jax.random.PRNGKey(3), 512, aux=0)
    pen = OrthogonalityPenalty(lambda s: ref.log_psi({}, s, 0), lam=4.0, ref_samples=ref_samples)
    st = ProductState(N, [RBM(alpha=2, scale=0.1), Jastrow(disp_index=lat.disp_index, n_classes=lat.n_classes)])
    cfg = VMCConfig(steps=120, n_samples=512, lr=0.05, diag_shift=1e-3, log_every=100)
    run = VMC(st, H, ExactSampler(N, 3), MinSR(lr=0.05, diag_shift=1e-3, max_step_norm=2.0), cfg, penalty=pen)
    hist = run.run(verbose=False)
    assert hist[-1]["ref_fidelity"] < 0.05
    assert run.energy > e[0] + 0.5 * (e[1] - e[0])


@pytest.mark.slow
def test_vit_square_4x4_metropolis():
    lat = presets.square(4); N = lat.N
    H = heisenberg(lat)
    e0 = diagonalize(H, k=1)[0][0]
    G = translation_group(lat, 0).times(point_group(lat, "C4", 4, 0, spin_flip=1))
    A = np.where(lat.site_cell.sum(axis=1) % 2 == 0)[0]                       # checkerboard sublattice
    st = ProductState(N, [vit_for(lat, d=16, n_layers=2, n_heads=2, d_head_mlp=32),
                          Jastrow(disp_index=lat.disp_index, n_classes=lat.n_classes), marshall_sign(A)], symmetry=G)
    cfg = VMCConfig(steps=150, n_sweeps=8, thin=2, n_burn=8, lr=0.02, lr_final=0.01, diag_shift=1e-3, log_every=50)
    run = VMC(st, H, exchange_sampler(lat, n_chains=256, p_long=0.2), MinSR(lr=0.02, diag_shift=1e-3, max_step_norm=1.0), cfg)
    run.run(verbose=False)
    assert abs(run.energy - e0) / abs(e0) < 3e-2      # ~1.9 % after 100 steps on a GPU with this small network
