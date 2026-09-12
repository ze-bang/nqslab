import numpy as np
import jax, jax.numpy as jnp

from nqslab.lattice import presets
from nqslab.sampler import exchange_sampler, flip_sampler, MetropolisSampler, MixedMove, ExchangeMove, FlipMove, ExactSampler
from nqslab.models import RBM, ProductState


def _state(N, seed=0):
    st = ProductState(N, [RBM(alpha=1, scale=0.3)])
    return st, st.init(jax.random.PRNGKey(seed))


def test_exchange_conserves_sz_and_matches_exact_distribution():
    lat = presets.chain(6); N = lat.N
    st, p = _state(N)
    smp = exchange_sampler(lat, n_chains=200, p_long=0.3)
    sample = smp.make_sampler(st.log_psi, 0)
    S, last, acc = sample(p, smp.init_state(jax.random.PRNGKey(1)), jax.random.PRNGKey(2), 60, 2, 20)
    S = np.asarray(S)
    assert np.all(S.sum(axis=1) == 0) and 0.05 < float(acc) < 1.0
    ex = ExactSampler(N, N // 2)
    prob = ex.probabilities(st.log_psi, p, aux=0)
    codes = (S == 1).astype(int) @ (1 << np.arange(N))
    hist = np.bincount(np.searchsorted(ex.codes, codes), minlength=len(ex.codes)) / len(S)
    assert np.abs(hist - prob).max() < 0.02


def test_flip_sampler_matches_exact_distribution():
    N = 5
    st, p = _state(N)
    smp = flip_sampler(N, n_chains=200)
    sample = smp.make_sampler(st.log_psi, 0)
    S, _, acc = sample(p, smp.init_state(jax.random.PRNGKey(1)), jax.random.PRNGKey(2), 60, 2, 20)
    S = np.asarray(S)
    assert len(set(S.sum(axis=1).tolist())) > 1
    ex = ExactSampler(N)
    prob = ex.probabilities(st.log_psi, p, aux=0)
    codes = (S == 1).astype(int) @ (1 << np.arange(N))
    hist = np.bincount(codes, minlength=2 ** N) / len(S)
    assert np.abs(hist - prob).max() < 0.02


def test_mixed_move():
    N = 6
    st, p = _state(N)
    move = MixedMove([ExchangeMove(np.array([[i, (i + 1) % N] for i in range(N)])), FlipMove()], [0.5, 0.5])
    smp = MetropolisSampler(N, move, n_chains=50)
    S, _, acc = smp.make_sampler(st.log_psi, 0)(p, smp.init_state(jax.random.PRNGKey(0), n_up=3), jax.random.PRNGKey(1), 10, 1, 2)
    assert np.asarray(S).shape == (500, N) and float(acc) > 0


def test_cluster_flip_move_preserves_a_constraint():
    """Flipping whole stars of a toric code keeps every plaquette satisfied, which single flips cannot."""
    from nqslab.sampler import ClusterFlipMove, MetropolisSampler
    from nqslab.lattice import Lattice
    lat = Lattice(np.eye(2), [[0.5, 0.0], [0.0, 0.5]], 3, name="square-links")
    stars, plaqs = [], []
    for cell in lat.cells:
        cell = np.asarray(cell)
        stars.append([lat.index_of(cell, 0)[0], lat.index_of(cell, 1)[0],
                      lat.index_of(cell - [1, 0], 0)[0], lat.index_of(cell - [0, 1], 1)[0]])
        plaqs.append([lat.index_of(cell, 0)[0], lat.index_of(cell, 1)[0],
                      lat.index_of(cell + [0, 1], 0)[0], lat.index_of(cell + [1, 0], 1)[0]])
    stars = np.array(stars); plaqs = np.array(plaqs)
    N = lat.N
    st = ProductState(N, [RBM(alpha=1, scale=0.02)])            # nearly flat, so moves are accepted
    p = st.init(jax.random.PRNGKey(0))
    smp = MetropolisSampler(N, ClusterFlipMove(stars), n_chains=16)
    s0 = jnp.ones((16, N), dtype=jnp.int8)                       # all up: every plaquette is +1
    assert np.all(np.prod(np.asarray(s0)[:, plaqs], axis=2) == 1)
    S, _, acc = smp.make_sampler(st.log_psi, 0)(p, s0, jax.random.PRNGKey(0), 20, 1, 5)
    S = np.asarray(S)
    assert np.all(np.prod(S[:, plaqs], axis=2) == 1)             # the constraint survives every move
    assert float(acc) > 0.5                                      # and a flat state accepts almost everything
    assert len(np.unique(S, axis=0)) > 32                        # so the chain explores the sector
