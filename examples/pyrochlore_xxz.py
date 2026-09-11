"""XXZ quantum spin ice on the 16-site pyrochlore cell with a transverse probe field: S^z is not
conserved, so the sampler mixes single flips with exchanges and the state lives in the full space.

    python examples/pyrochlore_xxz.py --Jpm -0.05 --hx 0.05
"""
import argparse
import numpy as np
from nqslab.lattice import presets
from nqslab.operator import xxz, field
from nqslab.ed import diagonalize
from nqslab.models import vit_for, Jastrow, ProductState, translation_group
from nqslab.sampler import MetropolisSampler, MixedMove, ExchangeMove, FlipMove
from nqslab.optim import MinSR
from nqslab.vmc import VMC, VMCConfig

ap = argparse.ArgumentParser()
ap.add_argument("--Jpm", type=float, default=-0.05); ap.add_argument("--hx", type=float, default=0.05)
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--cells", type=int, nargs=3, default=(1, 1, 1))
args = ap.parse_args()

lat = presets.pyrochlore(cubic_cells=args.cells)
H = xxz(lat, 1.0, args.Jpm) + field(lat.N, args.hx, "x")
G = translation_group(lat, 0)
psi = ProductState(lat.N, [vit_for(lat, d=32, n_layers=3), Jastrow(lat.disp_index, lat.n_classes)], symmetry=G)
move = MixedMove([ExchangeMove(lat.bonds["nn"].pairs, p_long=0.2), FlipMove()], [0.7, 0.3])
smp = MetropolisSampler(lat.N, move, n_chains=512)
cfg = VMCConfig(steps=args.steps, n_sweeps=8, thin=2, n_burn=8, lr=0.05, lr_final=0.01, diag_shift=1e-3, log_every=10)
run = VMC(psi, H, smp, MinSR(lr=0.05, diag_shift=1e-3, max_step_norm=2.0), cfg)
run.run()
if lat.N <= 16:
    e0 = diagonalize(H, k=1, full=True)[0][0]
    print(f"ED (full space): {e0:.6f}   VMC: {run.energy:.6f}   rel. error {(run.energy - e0) / abs(e0):.2e}")
