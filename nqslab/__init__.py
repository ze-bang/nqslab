"""nqslab: neural quantum states for lattice spin-1/2 models.

Layers (each usable on its own):

* :mod:`nqslab.lattice`     Bravais lattices with a basis on a torus in 1-3 dimensions, bonds with
                            windings, space-group permutations, displacement classes, presets
* :mod:`nqslab.operator`    sums of products of S+, S-, Sz, Sx, Sy; compiled connection rules
* :mod:`nqslab.sampler`     vectorised Metropolis chains with pluggable move sets; exact sampling
* :mod:`nqslab.models`      flax ansatze (factored-attention transformer, RBM, Jastrow, MLP),
                            products with fixed priors, symmetry projection
* :mod:`nqslab.optim`       per-sample Jacobians, minSR (plain and parameter-chunked), SR with CG,
                            optax first-order optimisers
* :mod:`nqslab.vmc`         local estimators of any operator, the VMC driver, penalties
* :mod:`nqslab.estimators`  overlaps, Renyi-2 swap, reduced density matrices, Chern number, MES
* :mod:`nqslab.ed`          sparse exact diagonalization from the same connection rule

Conventions: a configuration is an array of +-1 (s = 2 S^z), site i up <-> bit i set in the
integer code. ``log_psi(params, s, aux)`` maps a batch ``s`` of shape (B, N) to complex (B,).
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax as _jax

_jax.config.update("jax_enable_x64", True)
# TF32 matmuls (the GPU default) give ~1e-3 relative precision, too coarse for variational energies.
_jax.config.update("jax_default_matmul_precision", "highest")

from .lattice import Lattice, presets  # noqa: E402
from .operator import Operator, spin_ops, ConnectionRule, compile_operator  # noqa: E402

__version__ = "0.1.0"
__all__ = ["Lattice", "presets", "Operator", "spin_ops", "ConnectionRule", "compile_operator", "__version__"]
