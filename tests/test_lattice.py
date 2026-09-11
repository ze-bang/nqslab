import numpy as np
import pytest

from nqslab.lattice import presets, Lattice


@pytest.mark.parametrize("name,args,coord,nb", [
    ("chain", (8,), 2, 1), ("square", (4,), 4, 1), ("triangular", (3,), 6, 1), ("honeycomb", (3,), 3, 2),
    ("kagome", (2,), 4, 3), ("cubic", (2,), 6, 1), ("pyrochlore", (), 6, 4)])
def test_coordination(name, args, coord, nb):
    lat = presets.build(name, *args)
    assert lat.n_basis == nb
    b = lat.bonds["nn"]
    deg = np.bincount(np.concatenate([b.j, b.k]), minlength=lat.N)
    assert np.all(deg == coord)
    assert len(b) == lat.N * coord // 2


def test_tilted_torus_and_windings():
    lat = presets.triangular(N=12)
    assert lat.N == 12
    b = lat.bonds["nn"]
    assert b.winding.shape == (36, 2) and np.any(b.winding != 0)
    for a in range(len(b)):
        mu, nu, R = b.types[b.type[a]]
        idx, w = lat.index_of(lat.site_cell[b.j[a]] + np.asarray(R), nu)
        assert idx == b.k[a] and tuple(w) == tuple(b.winding[a])


def test_translations_form_group():
    for lat in (presets.triangular(N=12), presets.honeycomb(3), presets.pyrochlore()):
        T = lat.translations
        keys = {tuple(p) for p in T}
        assert len(keys) == lat.n_cells
        for p in T:
            for q in T:
                assert tuple(q[p]) in keys
        # translations preserve the bond structure
        bonds = {frozenset((int(j), int(k))) for j, k in lat.bonds["nn"].pairs}
        for p in T:
            assert {frozenset((int(p[j]), int(p[k]))) for j, k in lat.bonds["nn"].pairs} == bonds


def test_point_operations_preserve_bonds():
    cases = [(presets.square(4), "C4", 4), (presets.triangular(N=12), "C6", 6), (presets.honeycomb(3), "C6", 6),
             (presets.kagome(2), "C6", 6), (presets.cubic(2), "C4z", 4)]
    for lat, name, order in cases:
        p = lat.point_ops()[name]
        bonds = {frozenset((int(j), int(k))) for j, k in lat.bonds["nn"].pairs}
        assert {frozenset((int(p[j]), int(p[k]))) for j, k in lat.bonds["nn"].pairs} == bonds
        q = np.arange(lat.N)
        for _ in range(order):
            q = p[q]
        assert np.all(q == np.arange(lat.N))


def test_incompatible_torus_has_no_c6():
    lat = presets.triangular(((4, 0), (0, 3)))
    assert "C6" not in lat.point_ops()


def test_disp_index_is_translation_invariant():
    lat = presets.honeycomb(3)
    D = lat.disp_index
    assert D.max() < lat.n_classes
    for p in lat.translations:
        assert np.all(D[np.ix_(p, p)] == D)


def test_triangles_triangular():
    lat = presets.triangular(N=12)
    tri = lat.triangles()
    assert tri.shape == (24, 3)
    disp = lat.bond_displacement("nn")
    # every triangle is counter-clockwise: check with the cartesian minimum-image positions
    for (i, j, k) in tri:
        d1 = lat.positions[j] - lat.positions[i]; d2 = lat.positions[k] - lat.positions[j]
        # minimum image
        for d in (d1, d2):
            f = d @ np.linalg.inv(lat.torus_vectors); f -= np.round(f); d[:] = f @ lat.torus_vectors
        assert d1[0] * d2[1] - d1[1] * d2[0] > 0


def _unoriented(types):
    return {frozenset(((mu, R), (nu, tuple(-x for x in R)))) for mu, nu, R in types}


def test_neighbor_shells_match_presets():
    lat = presets.square(4)
    shells = lat.neighbor_shells(2)
    assert _unoriented(shells[0]) == _unoriented(lat.bond_types["nn"])
    assert _unoriented(shells[1]) == _unoriented(lat.bond_types["nnn"])


def test_distance_and_momenta():
    lat = presets.square(4)
    assert np.isclose(lat.distance[0, 1], 1.0) and np.isclose(lat.distance[0, 2], 2.0)
    k = lat.momenta()
    assert k.shape == (16, 2)
    # every momentum is compatible with the torus: exp(i k.T) = 1
    assert np.allclose(np.exp(1j * (k @ lat.torus_vectors.T)), 1.0)
