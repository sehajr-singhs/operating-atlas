"""
S1: can the geometric routing coordinates be recovered from finite data?

We simulate a stochastic damped pendulum with state-dependent diffusion, for
which the drift and diffusion are known in closed form. The ground-truth
routing coordinates are computed by running the *exact* drift and diffusion
through the same GeometricFeatures construction used at inference, so the
comparison isolates estimation error rather than discretisation error.

Reported: rank and linear correlation between estimated and true (R, kappa),
relative error, and the sample-size sweep that says how much telemetry a
practitioner actually needs.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn

from manifold import LatentSDE, GeometricFeatures

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------
# ground-truth system
# ----------------------------------------------------------------------------

A_DRIVE, DAMP = 0.35, 0.25


def true_mu(z):
    """Damped driven pendulum: strongly curved flow, non-uniform stretching."""
    return torch.stack([z[..., 1],
                        -torch.sin(z[..., 0]) - DAMP * z[..., 1] + A_DRIVE], -1)


def true_L(z):
    """Lower-triangular Cholesky factor of V(z); state dependent and coupled."""
    s1 = 0.25 * (1.0 + 0.6 * torch.cos(z[..., 0]))
    s2 = 0.20 * (1.0 + 0.8 * z[..., 1] ** 2 / (1.0 + z[..., 1] ** 2))
    c = 0.10 * torch.tanh(z[..., 0])
    zero = torch.zeros_like(s1)
    return torch.stack([torch.stack([s1, zero], -1),
                        torch.stack([c, s2], -1)], -2)


def true_V(z):
    L = true_L(z)
    return L @ L.transpose(-1, -2)


class OracleSDE(LatentSDE):
    """Wraps the analytic drift/diffusion in the LatentSDE interface so the same
    GeometricFeatures code produces the ground-truth coordinates."""

    def __init__(self):
        super().__init__(k=2, width=8, depth=1)

    def mu(self, z):            # type: ignore[override]
        return true_mu(z)

    def L(self, z):             # type: ignore[override]
        return true_L(z)

    def V(self, z):             # type: ignore[override]
        return true_V(z)


def simulate(n_traj, n_steps, dt, seed=0):
    """Euler-Maruyama rollouts, returning (z_t, dz_t) transition pairs."""
    g = torch.Generator().manual_seed(seed)
    z = torch.stack([torch.rand(n_traj, generator=g) * 2 * np.pi - np.pi,
                     torch.randn(n_traj, generator=g) * 0.8], -1)
    Z, DZ = [], []
    for _ in range(n_steps):
        L = true_L(z)
        dw = torch.randn(n_traj, 2, generator=g) * dt ** 0.5
        dz = true_mu(z) * dt + torch.einsum('nij,nj->ni', L, dw)
        Z.append(z.clone())
        DZ.append(dz.clone())
        z = z + dz
        z[..., 0] = (z[..., 0] + np.pi) % (2 * np.pi) - np.pi   # wrap angle
    return torch.cat(Z), torch.cat(DZ)


# ----------------------------------------------------------------------------
# fit + evaluate
# ----------------------------------------------------------------------------

def fit_sde(Z, DZ, dt, epochs=300, width=96, seed=0, verbose=False):
    torch.manual_seed(seed)
    sde = LatentSDE(k=2, width=width, depth=3)
    opt = torch.optim.Adam(sde.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = Z.shape[0]
    bs = min(4096, n)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = sde.nll(Z[idx], DZ[idx], dt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and ep % 50 == 0:
            print(f'    ep {ep:4d}  nll {tot / n:+.4f}')
    return sde


def corr(a, b):
    a, b = a.numpy(), b.numpy()
    pear = float(np.corrcoef(a, b)[0, 1])
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    spear = float(np.corrcoef(ra, rb)[0, 1])
    return pear, spear


def main():
    dt = 0.02
    tau = 0.30
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-2.6, 2.6, 22), torch.linspace(-1.9, 1.9, 22), indexing='ij'), -1).reshape(-1, 2)

    print('computing ground-truth geometric coordinates from the analytic SDE ...')
    t0 = time.time()
    oracle = GeometricFeatures(OracleSDE(), tau=tau)
    F_true = oracle(grid)
    print(f'  {grid.shape[0]} points in {time.time() - t0:.1f}s')
    print(f'  R    range [{F_true[:,0].min():+.3f}, {F_true[:,0].max():+.3f}]  '
          f'std {F_true[:,0].std():.3f}')
    print(f'  kappa range [{F_true[:,1].min():+.3f}, {F_true[:,1].max():+.3f}]  '
          f'std {F_true[:,1].std():.3f}')

    results = {'tau': tau, 'dt': dt, 'sweep': []}
    for n_traj, n_steps in [(40, 250), (100, 500), (250, 800), (600, 1200)]:
        Z, DZ = simulate(n_traj, n_steps, dt, seed=1)
        n = Z.shape[0]
        rows = []
        for seed in range(3):
            sde = fit_sde(Z, DZ, dt, epochs=300, seed=seed)
            F_hat = GeometricFeatures(sde, tau=tau)(grid)
            # drift / diffusion recovery
            with torch.no_grad():
                mu_err = ((sde.mu(grid) - true_mu(grid)).norm(dim=-1)
                          / true_mu(grid).norm(dim=-1).clamp_min(1e-8)).median().item()
                V_err = ((sde.V(grid) - true_V(grid)).flatten(1).norm(dim=-1)
                         / true_V(grid).flatten(1).norm(dim=-1)).median().item()
            pR, sR = corr(F_hat[:, 0], F_true[:, 0])
            pK, sK = corr(F_hat[:, 1], F_true[:, 1])
            rows.append(dict(seed=seed, mu_rel=mu_err, V_rel=V_err,
                             R_pearson=pR, R_spearman=sR,
                             kappa_pearson=pK, kappa_spearman=sK))
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'seed'}
        agg_sd = {k + '_sd': float(np.std([r[k] for r in rows])) for k in rows[0] if k != 'seed'}
        entry = dict(n_transitions=n, **agg, **agg_sd)
        results['sweep'].append(entry)
        print(f"n={n:8d}  mu_rel={agg['mu_rel']:.3f}  V_rel={agg['V_rel']:.3f}  "
              f"R_spear={agg['R_spearman']:.3f}+-{agg_sd['R_spearman_sd']:.3f}  "
              f"kappa_spear={agg['kappa_spearman']:.3f}+-{agg_sd['kappa_spearman_sd']:.3f}")

    with open('../results/s1_recovery.json', 'w') as f:
        json.dump(results, f, indent=2)
    torch.save({'grid': grid, 'F_true': F_true}, '../results/s1_truth.pt')
    print('\nwrote results/s1_recovery.json')


if __name__ == '__main__':
    main()
