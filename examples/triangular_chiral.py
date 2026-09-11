"""J1-J2-Jchi on a C6 triangular torus: the chiral spin liquid model, with a C6-projected transformer.

    python examples/triangular_chiral.py --N 12 --J2 0.1 --Jchi -0.3
"""
import argparse
from nqslab.lattice import presets
from nqslab.operator import heisenberg, chirality
from nqslab.ed import diagonalize
from nqslab.models import vit_for, Jastrow, ProductState, point_group
from nqslab.sampler import exchange_sampler
from nqslab.optim import MinSR
from nqslab.vmc import VMC, VMCConfig

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=12); ap.add_argument("--J2", type=float, default=0.1)
ap.add_argument("--Jchi", type=float, default=-0.3); ap.add_argument("--steps", type=int, default=300)
args = ap.parse_args()

lat = presets.triangular(N=args.N)
H = heisenberg(lat) + heisenberg(lat, args.J2, "nnn") + chirality(lat, args.Jchi)
G = point_group(lat, "C6", 6, 0, spin_flip=1)
psi = ProductState(lat.N, [vit_for(lat, d=32, n_layers=4), Jastrow(lat.disp_index, lat.n_classes)], symmetry=G)
cfg = VMCConfig(steps=args.steps, n_sweeps=8, thin=2, n_burn=8, lr=0.05, lr_final=0.01, diag_shift=1e-3,
                diag_shift_final=1e-4, compact_fraction=0.75, log_every=10)
run = VMC(psi, H, exchange_sampler(lat, n_chains=512, p_long=0.2), MinSR(lr=0.05, diag_shift=1e-3, max_step_norm=2.0), cfg)
run.run()
if lat.N <= 24:
    e0 = diagonalize(H, k=1)[0][0]
    print(f"ED: {e0:.6f}   VMC: {run.energy:.6f}   rel. error {(run.energy - e0) / abs(e0):.2e}")
