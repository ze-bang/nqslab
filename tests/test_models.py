import numpy as np
import jax, jax.numpy as jnp

from nqslab.lattice import presets
from nqslab.models import (vit_for, RBM, Jastrow, MLP, Fixed, ProductState, SymmetryGroup, translation_group,
                           point_group)
from nqslab.ed import sz_basis


def _check_projection(st, params, basis, G):
    """psi_G(g^-1 s) = chi(g) psi_G(s) for every g, with an absolute tolerance set by the unprojected
    amplitudes over the orbit (an irrep may annihilate a configuration, where ratios are meaningless)."""
    s = jnp.asarray(basis[:6])
    st0 = ProductState(st.N, st.components)                      # same parameters, no projection
    orbit = G.apply_all(s).reshape(G.order * s.shape[0], st.N)
    scale = np.exp(np.asarray(st0.log_psi(params, orbit, 0)).real.reshape(G.order, -1).max(axis=0))
    psi = np.exp(np.asarray(st.log_psi(params, s, 0)))
    for g in range(G.order):
        sg = s[:, G.perms[g]] * G.flips[g]
        psi_g = np.exp(np.asarray(st.log_psi(params, sg, 0)))
        assert np.all(np.abs(psi_g - G.chars[g] * psi) < 1e-9 * scale), g


def test_vit_is_translation_invariant_on_multi_sublattice_lattice():
    lat = presets.honeycomb(2)
    net = vit_for(lat, d=8, n_layers=1, n_heads=2, d_head_mlp=8)
    st = ProductState(lat.N, [net])
    p = st.init(jax.random.PRNGKey(0))
    basis, _ = sz_basis(lat.N)
    s = jnp.asarray(basis[:4])
    lp = st.log_psi(p, s, 0)
    for t in lat.translations:
        assert np.allclose(np.asarray(st.log_psi(p, s[:, t], 0)), np.asarray(lp), atol=1e-6)


def test_projection_momentum_point_group_and_flip():
    lat = presets.square(3)
    basis, _ = sz_basis(lat.N, 4)
    G = translation_group(lat, 0).times(point_group(lat, "C4", 4, 1, spin_flip=-1))
    assert G.check_group() and G.order == 9 * 4 * 2
    import pytest
    with pytest.raises(ValueError):          # a nonzero momentum is not C4 invariant on this torus: no 1D irrep
        translation_group(lat, 2).times(point_group(lat, "C4", 4, 0))
    st = ProductState(lat.N, [RBM(alpha=1, scale=0.2), Jastrow(disp_index=lat.disp_index, n_classes=lat.n_classes)],
                      symmetry=G)
    _check_projection(st, st.init(jax.random.PRNGKey(0)), basis, G)


def test_inconsistent_characters_raise():
    lat = presets.square(3)
    p = lat.point_ops()["C4"]
    import pytest
    with pytest.raises(ValueError):
        SymmetryGroup.from_generators([(p, 1, np.exp(2j * np.pi / 3))], lat.N)


def test_fixed_component_and_aux_grid():
    lat = presets.chain(6); N = lat.N
    fixed = Fixed(lambda s, aux: 0.1 * jnp.sum(s, axis=1) + 0j)
    grid = np.array([[0.0, 0.0], [0.5, 0.2]])
    feats = lambda th: jnp.concatenate([jnp.cos(th), jnp.sin(th)], axis=-1)
    net = vit_for(lat, d=8, n_layers=1, n_heads=1, d_head_mlp=8, n_aux=4)
    st = ProductState(N, [net, fixed, MLP((8,))], aux_grid=grid, aux_features=feats)
    p = st.init(jax.random.PRNGKey(0))
    assert set(p) == {"c0", "c2"} and st.K == 2
    s = jnp.asarray(sz_basis(N)[0][:3])
    a = st.log_psi(p, s, 0); b = st.log_psi(p, s, 1); c = st.log_psi(p, s, jnp.array([0, 1, 0]))
    assert not np.allclose(a, b) and np.allclose(np.asarray(c)[[0, 2]], np.asarray(a)[[0, 2]]) and np.isclose(np.asarray(c)[1], np.asarray(b)[1])


def test_loop_feature_is_exactly_the_loop_parities():
    """Each loop contributes c_k times the product of its spins, with unequal lengths handled."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from nqslab.models import LoopFeature

    loops = ((0, 1, 2, 3), (1, 4), (0, 2, 4))
    m = LoopFeature(loops=loops, scale=0.4)
    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.choice([-1, 1], size=(16, 5)).astype(np.int8))
    v = m.init(jax.random.PRNGKey(1), s)
    c = np.asarray(v["params"]["c_re"] + 1j * v["params"]["c_im"])
    w = np.stack([np.prod(np.asarray(s)[:, list(l)], axis=1) for l in loops], axis=1)
    assert np.allclose(np.asarray(m.apply(v, s)), w @ c)
    # the score function of c_k is the loop itself, which is what makes the direction representable
    g = jax.jacrev(lambda p: m.apply(p, s).real)(v)["params"]["c_re"]
    assert np.allclose(np.asarray(g), w)


def test_loop_feature_with_no_loops_is_inert():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nqslab.models import LoopFeature

    m = LoopFeature(loops=())
    s = jnp.asarray(np.ones((4, 6), dtype=np.int8))
    v = m.init(jax.random.PRNGKey(0), s)
    assert np.allclose(np.asarray(m.apply(v, s)), 0.0)
