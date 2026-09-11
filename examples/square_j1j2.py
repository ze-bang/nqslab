"""J1-J2 Heisenberg on the square torus with a symmetry-projected transformer x Jastrow state.

    python examples/square_j1j2.py --L 4 --J2 0.5 --steps 300
"""
import argparse
import numpy as np
from nqslab.lattice import presets
from nqslab.operator import heisenberg
from nqslab.ed import diagonalize
from nqslab.models import vit_for, Jastrow, ProductState, translation_group, point_group
from nqslab.sampler import exchange_sampler
from nqslab.optim import MinSR
from nqslab.vmc import VMC, VMCConfig

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=4); ap.add_argument("--J2", type=float, default=0.5)
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--d", type=int, default=32)
ap.add_argument("--layers", type=int, default=4); ap.add_argument("--chains", type=int, default=512)
args = ap.parse_args()

lat = presets.square(args.L)
H = heisenberg(lat, 1.0) + heisenberg(lat, args.J2, "nnn")
G = translation_group(lat, 0).times(point_group(lat, "C4", 4, 0, spin_flip=1))
psi = ProductState(lat.N, [vit_for(lat, d=args.d, n_layers=args.layers), Jastrow(lat.disp_index, lat.n_classes)], symmetry=G)
cfg = VMCConfig(steps=args.steps, n_sweeps=8, thin=2, n_burn=8, lr=0.05, lr_final=0.01, diag_shift=1e-3,
                diag_shift_final=1e-4, log_every=10, out_dir=f"runs/square_L{args.L}_J2{args.J2}")
run = VMC(psi, H, exchange_sampler(lat, n_chains=args.chains, p_long=0.2), MinSR(lr=0.05, diag_shift=1e-3, max_step_norm=2.0), cfg)
run.run()
if lat.N <= 20:
    e0 = diagonalize(H, k=1)[0][0]
    print(f"ED: {e0:.6f}   VMC: {run.energy:.6f}   rel. error {(run.energy - e0) / abs(e0):.2e}")
