# nqslab

Neural quantum states for lattice spin-1/2 models in JAX. One package, seven layers, each usable on
its own:

| layer | what it gives you |
|---|---|
| `nqslab.lattice` | Bravais lattice + basis on any torus in 1-3D (chain, square, triangular, honeycomb, kagome, cubic, pyrochlore presets); bonds with windings for twisted boundaries; translations and point operations as site permutations; displacement classes for translation-invariant kernels; oriented triangles |
| `nqslab.operator` | sums of products of S+, S-, Sz, Sx, Sy with complex coefficients; algebra, Hermitian conjugate, dense matrices; `compile_operator` turns any operator into a fixed-shape connection rule (all s' with <s'\|O\|s> != 0) usable under `jit`; builders for Heisenberg, XXZ, scalar chirality, fields |
| `nqslab.sampler` | vectorised Metropolis with pluggable moves (exchange, single flip, mixtures) and exact sampling for small N |
| `nqslab.models` | factored-attention transformer (translation equivariant on any lattice with a basis), RBM, Jastrow pair kernels, MLP; `ProductState` composes learnable and fixed factors, conditions on an auxiliary grid (e.g. twist angles) and projects onto a symmetry irrep (momentum x point group x spin flip) |
| `nqslab.optim` | per-sample Jacobians; minSR (kernel solve), parameter-chunked minSR, SR with conjugate gradient, optax first-order optimisers |
| `nqslab.vmc` | local estimator of any operator (Hermitian or not), the VMC driver with schedules, checkpoints and penalties for excited states, expectation values, structure factors |
| `nqslab.estimators` | overlaps and fidelities, Renyi-2 swap and Kitaev-Preskill / Levin-Wen regions, reduced density matrices and the modular commutator, many-body Chern number from twist plaquettes, minimally entangled states |
| `nqslab.ed` | sparse exact diagonalization from the same connection rule (agrees with VMC by construction), adapter to the QED toolkit for large clusters |

```python
from nqslab.lattice import presets
from nqslab.operator import heisenberg
from nqslab.models import vit_for, Jastrow, ProductState, translation_group, point_group
from nqslab.sampler import exchange_sampler
from nqslab.optim import MinSR
from nqslab.vmc import VMC, VMCConfig

lat = presets.square(4)
H = heisenberg(lat) + heisenberg(lat, 0.5, "nnn")                 # J1-J2 on the 4x4 torus
G = translation_group(lat, 0).times(point_group(lat, "C4", 4, 0, spin_flip=1))
psi = ProductState(lat.N, [vit_for(lat, d=32, n_layers=4), Jastrow(lat.disp_index, lat.n_classes)], symmetry=G)
run = VMC(psi, H, exchange_sampler(lat, n_chains=512), MinSR(lr=0.05), VMCConfig(steps=500))
run.run()
```

## Install

```bash
pip install -e ".[test]"            # CPU JAX
pip install -e ".[cuda,test]"       # CUDA 12
pytest -q tests -m "not slow"       # ~2 min on CPU
```

## Conventions

* A configuration is an array of +-1 (s = 2 S^z); site i up <-> bit i of the integer code.
* `log_psi(params, s, aux)` maps a batch (B, N) to complex (B,); parameters are real, so every
  optimiser works on one flat real vector.
* `ConnectionRule.amplitudes(s)` returns <s'|O|s>. The local estimator of O uses
  <s|O|s'> = conj(<s'|O^dagger|s>), so non-Hermitian operators are estimated correctly.
* A twist theta multiplies S+_j S-_k on a bond that winds w times around the torus by exp(i theta.w).
* Triangles are stored counter-clockwise; `chirality(lat, Jchi)` is Jchi sum S_i.(S_j x S_k).
* Symmetry projection acts on the full product of factors (projecting one factor alone does not
  give the product the quantum number).
* Matmuls run at full precision (TF32 is disabled at import: 1e-3 relative error is too coarse for
  variational energies).

## Downstream projects

* [nqs_csl](https://github.com/ze-bang/nqs_csl): the triangular-lattice chiral spin liquid study
  (Laughlin and parton priors, bake-off configurations, the learning course).
