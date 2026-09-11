# Design notes

## Why a connection rule compiler

Every VMC quantity is a sum over the configurations connected to a sample:
E_loc(s) = sum_{s'} <s|H|s'> psi(s')/psi(s). Hand-deriving the list of s' and their amplitudes for
each model is where the mistakes live (the conjugate, the twist phase orientation, the sign of a
chirality term). `compile_operator` derives it once from the operator's term list:

1. every term is expanded into products of S+, S-, Sz (Sx, Sy split into two terms);
2. on each site the product of the factors is reduced to one 2x2 matrix, which is either diagonal,
   purely raising or purely lowering (products of single-entry matrices stay single-entry);
3. terms with the same *flip set* are grouped; a group is one connected configuration s' = s with
   the flip set toggled, and its amplitude is a sum of products of per-site table lookups.

All arrays have fixed shapes (terms padded with identity factors), so the rule is jitted and
vmapped; the number of connected configurations is the number of distinct flip sets, which also
merges bonds that coincide on small tori. The sparse ED matrix is built from the same rule, so ED
and VMC agree by construction, and the QED adapter sends the same simplified term list.

The estimator of an operator O evaluates <s|O|s'> = conj(<s'|O^dagger|s>) so non-Hermitian
correlators are estimated correctly; for a Hermitian Hamiltonian this is a no-op.

## Why displacement classes

Translation-invariant pair kernels and the factored attention need a label for "the pair (i, j)
up to translation". On a lattice with a basis that label is (mu_i, mu_j, cell_i - cell_j) reduced
to the torus, `lattice.disp_index`, with `n_basis^2 * n_cells` classes. The transformer's kernel is
a table over classes and every layer is exactly translation equivariant; sum pooling makes the
output invariant. Point-group and spin-flip quantum numbers are imposed by projection on the full
product state.

## Cost model

Everything is proportional to network forward passes: one per Metropolis proposal, one per
connected configuration in the local energy (compacted to allowed flips), two backward passes per
sample for the Jacobian, all multiplied by the symmetry-group order. `log_psi_fn` carries that
order as `cost` so the local estimator sizes its batches in network evaluations. minSR solves a
(2 Ns x 2 Ns) system and is exact when P > 2 Ns; `ChunkedMinSR` keeps the Jacobian in host memory
in sample chunks when 2 Ns x P does not fit on the device.

## What is deliberately not here (yet)

Local dimensions other than 2 (spin-S, bosons), fermions (Slater/backflow determinants and
Jordan-Wigner signs), autoregressive samplers, and finite temperature. The Operator layer and
the sampler moves are the extension points for the first three.
