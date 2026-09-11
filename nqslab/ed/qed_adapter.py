"""Adapter to the QED exact-diagonalization toolkit (github.com/ze-bang/QED_Spin).

QED's builder accepts arbitrary products of S+, S-, Sz on sites with complex coefficients, which
is exactly the term list of an :class:`~nqslab.operator.Operator` after :meth:`simplify`. QED is a
separate build; this module is importable without it and raises a clear error when ``qed`` is
unavailable. Eigenvectors come back in the computational basis (bit i = site i up), the same
convention as :mod:`nqslab.ed.sparse`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..operator import Operator

_TOKEN = {"sp": "+", "sm": "-", "sz": "z"}


def term_list(op: Operator) -> List[Tuple[Tuple[str, ...], Tuple[int, ...], complex]]:
    """All terms as (QED tokens, sites, coefficient), merged and expanded into +/-/z products."""
    out = []
    for t in op.simplify().terms:
        if len(t.ops) == 0:
            continue                        # identity terms: constant shift, add back by hand if needed
        out.append((tuple(_TOKEN[o] for o in t.ops), tuple(int(s) for s in t.sites), complex(t.coeff)))
    return out


def constant_term(op: Operator) -> complex:
    return sum((t.coeff for t in op.simplify().terms if len(t.ops) == 0), 0j)


def build_qed_operator(op: Operator, n_up: Optional[int] = None):
    try:
        import qed  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("qed is not importable in this environment; use its virtualenv, or nqslab.ed.sparse for N <= ~26") from e
    H = qed.hamiltonian.Hamiltonian(num_sites=op.N, n_up=n_up)
    for ops, sites, coeff in term_list(op):
        H.add(list(ops), list(sites), complex(coeff))
    return H.build()


def solve_qed(op: Operator, n_eigs: int = 4, n_up: Optional[int] = None, **kw) -> Dict[str, Any]:
    import qed  # type: ignore
    n_up = op.N // 2 if n_up is None else n_up
    built = build_qed_operator(op, n_up=n_up)
    res = qed.solve(built, num_eigenvalues=n_eigs, compute_eigenvectors=True, sz=n_up, **kw)
    out = dict(eigenvalues=np.array(res.eigenvalues) + constant_term(op).real)
    vecs = getattr(res, "eigenvectors", None)
    if vecs:
        out["eigenvectors"] = [np.asarray(v) for v in vecs]
    out["eigenvectors_path"] = getattr(res, "eigenvectors_path", "")
    return out
