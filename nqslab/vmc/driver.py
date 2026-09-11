"""The VMC driver: sample -> local energies -> Jacobian -> optimiser step, with schedules, logging and checkpoints."""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..operator import Operator, make_local_estimator
from ..optim import JacobianFn, flatten, make_logpsi_flat, MinSR
from ..sampler import MetropolisSampler, ExactSampler
from ..models import ProductState
from .penalty import OrthogonalityPenalty


@dataclass
class VMCConfig:
    steps: int = 500
    n_sweeps: int = 16                 # sweeps per step (Metropolis) ; samples = n_sweeps // thin * n_chains
    thin: int = 4
    n_burn: int = 8
    n_samples: int = 2048              # samples per step for an ExactSampler
    lr: float = 0.02
    lr_final: Optional[float] = None
    diag_shift: float = 1e-3
    diag_shift_final: Optional[float] = None
    max_step_norm: float = 1.0
    jac_chunk: int = 256
    eloc_target_configs: int = 65536
    compact_fraction: float = 1.0
    seed: int = 0
    out_dir: Optional[str] = None
    log_every: int = 10
    checkpoint_every: int = 100


class VMC:
    """Ground-state search for ``state`` and ``hamiltonian`` (one Operator, or one per aux point of the state)."""

    def __init__(self, state: ProductState, hamiltonian: Union[Operator, Sequence[Operator]],
                 sampler: Union[MetropolisSampler, ExactSampler], optimizer=None, cfg: Optional[VMCConfig] = None,
                 penalty: Optional[OrthogonalityPenalty] = None, params: Optional[Dict] = None):
        self.state = state
        self.cfg = cfg or VMCConfig()
        self.K = state.K
        hams = list(hamiltonian) if isinstance(hamiltonian, (list, tuple)) else [hamiltonian] * self.K
        assert len(hams) == self.K, "one Hamiltonian per aux point"
        self.hams = hams
        self.sampler = sampler
        self.key = jax.random.PRNGKey(self.cfg.seed)
        self.key, k = jax.random.split(self.key)
        self.params = state.init(k) if params is None else params
        self.theta, self.unravel = flatten(self.params)
        self.exact = isinstance(sampler, ExactSampler)
        if not self.exact:
            self.sample_fns = [sampler.make_sampler(state.log_psi, k) for k in range(self.K)]
            self.states = [sampler.init_state(jax.random.PRNGKey(self.cfg.seed + 100 + k)) for k in range(self.K)]
        self.eloc_fns = [make_local_estimator(h, state.log_psi_fn(k), target_configs=self.cfg.eloc_target_configs,
                                              compact_fraction=self.cfg.compact_fraction) for k, h in enumerate(hams)]
        self.jac_fns = [JacobianFn(make_logpsi_flat(state, self.unravel, k), self.cfg.jac_chunk) for k in range(self.K)]
        self.opt = optimizer if optimizer is not None else MinSR(lr=self.cfg.lr, diag_shift=self.cfg.diag_shift,
                                                                  max_step_norm=self.cfg.max_step_norm)
        self.penalty = penalty
        self.history: List[Dict[str, float]] = []
        if self.cfg.out_dir:
            os.makedirs(self.cfg.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ pieces
    def schedule(self, it: int, steps: int):
        c = self.cfg
        frac = it / max(1, steps - 1)
        lr_final = c.lr_final if c.lr_final is not None else c.lr
        self.opt.lr = c.lr + (lr_final - c.lr) * 0.5 * (1 - np.cos(np.pi * frac))
        sf = c.diag_shift_final if c.diag_shift_final is not None else c.diag_shift
        self.opt.diag_shift = float(np.exp(np.log(c.diag_shift) + (np.log(sf) - np.log(c.diag_shift)) * frac))

    def sample(self, k: int, it: int = 0):
        c = self.cfg
        params = self.unravel(self.theta)
        self.key, ks = jax.random.split(self.key)
        if self.exact:
            S = self.sampler.sample(self.state.log_psi, params, ks, c.n_samples, aux=k)
            return S, 1.0
        S, self.states[k], acc = self.sample_fns[k](params, self.states[k], ks, c.n_sweeps, c.thin, c.n_burn if it < self.K else 0)
        return S, float(acc)

    def step(self, it: int = 0, steps: Optional[int] = None) -> Dict[str, float]:
        self.schedule(it, self.cfg.steps if steps is None else steps)
        k = it % self.K
        params = self.unravel(self.theta)
        S, acc = self.sample(k, it)
        E = self.eloc_fns[k](params, S)
        extra = None; fid = None
        if self.penalty is not None and k == 0:
            extra, fid = self.penalty.local_term(lambda s: self.state.log_psi(params, s, 0), S)
        d, info = self.opt.step(self.jac_fns[k], self.theta, S, E, extra_local=extra)
        self.theta = self.theta + d
        rec = dict(step=it, k=k, acc=acc, n_samples=int(S.shape[0]), **info)
        if fid is not None:
            rec["ref_fidelity"] = fid
        return rec

    def run(self, steps: Optional[int] = None, callback: Optional[Callable[[Dict], None]] = None, verbose: bool = True):
        c = self.cfg
        steps = c.steps if steps is None else steps
        t0 = time.time()
        for it in range(steps):
            rec = self.step(it, steps)
            rec["time"] = time.time() - t0
            self.history.append(rec)
            if callback is not None:
                callback(rec)
            if verbose and (it % c.log_every == 0 or it == steps - 1):
                extra = f"  fid_ref={rec['ref_fidelity']:.3f}" if "ref_fidelity" in rec else ""
                print(f"[{rec['time']:7.1f}s] step {it:5d} k={rec['k']} E={rec['E']:.6f} +- {rec['E_err']:.6f} "
                      f"acc={rec['acc']:.2f} |d|={rec['step_norm']:.3f} shift={rec['shift']:.1e}{extra}", flush=True)
                if c.out_dir:
                    with open(os.path.join(c.out_dir, "history.json"), "w") as fh:
                        json.dump(self.history, fh)
            if c.out_dir and c.checkpoint_every and (it + 1) % c.checkpoint_every == 0:
                self.save()
        if c.out_dir:
            self.save()
        return self.history

    # ------------------------------------------------------------------ io
    def save(self, path: Optional[str] = None):
        path = path or os.path.join(self.cfg.out_dir, "checkpoint.pkl")
        with open(path, "wb") as fh:
            pickle.dump(dict(config=asdict(self.cfg), theta=np.asarray(self.theta), history=self.history), fh)

    def load(self, path: str):
        with open(path, "rb") as fh:
            ck = pickle.load(fh)
        self.theta = jnp.asarray(ck["theta"]); self.history = list(ck.get("history", []))
        return ck

    # ------------------------------------------------------------------ evaluation helpers
    @property
    def energy(self) -> float:
        return self.history[-1]["E"] if self.history else float("nan")

    def log_psi_numpy(self, k: int = 0, batch: int = 2048) -> Callable:
        params = self.unravel(self.theta)
        def f(s):
            out = []
            for a in range(0, len(s), batch):
                out.append(np.asarray(self.state.log_psi(params, jnp.asarray(np.asarray(s[a:a + batch]).astype(np.int8)), k)))
            return np.concatenate(out)
        return f

    def draw_samples(self, k: int = 0, n_sweeps: int = 16, thin: int = 4, seed: int = 123) -> np.ndarray:
        params = self.unravel(self.theta)
        if self.exact:
            return np.asarray(self.sampler.sample(self.state.log_psi, params, jax.random.PRNGKey(seed), n_sweeps * self.cfg.n_samples // max(1, thin), aux=k))
        S, _, _ = self.sample_fns[k](params, self.states[k], jax.random.PRNGKey(seed), n_sweeps, thin, 0)
        return np.asarray(S)
