"""
Validation of the two routing invariants, plus a cost benchmark.

Part A (analytic). Push an analytic SDE forward through a diffeomorphism f and
check that (R, Pe) computed in the new coordinates at f(z) equals (R, Pe)
computed in the old coordinates at z. This pins the tensor algebra.

Part B (estimated). Simulate trajectories, relabel the samples by f, refit the
neural SDE from scratch in the new coordinates, and check that the *estimated*
invariants still agree. This is the claim the routing argument actually rests
on: invariance has to survive estimation, not just algebra.

Part C. Wall-clock cost of the invariants against the latent dimension.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from manifold import LatentSDE, Invariants
from geometry import scalar_curvature

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------
# an analytic 2-d test system: damped driven pendulum, state-dependent noise
# ----------------------------------------------------------------------------

DAMP, DRIVE = 0.25, 0.35


def sig_z(z):
    """Sigma(z), lower triangular, state dependent and cross-coupled."""
    s1 = 0.30 * (1.0 + 0.6 * torch.cos(z[0]))
    s2 = 0.22 * (1.0 + 0.8 * z[1] ** 2 / (1.0 + z[1] ** 2))
    c = 0.12 * torch.tanh(z[0])
    zero = torch.zeros((), dtype=z.dtype)
    return torch.stack([torch.stack([s1, zero]), torch.stack([c, s2])])


def drift_strat_z(z):
    """Stratonovich drift of the test system (defined directly in this form)."""
    return torch.stack([z[1], -torch.sin(z[0]) - DAMP * z[1] + DRIVE])


def diffeo(z):
    """A smooth, non-affine, coordinate-coupling relabelling of the state.

    Triangular in (z0, z1) and strictly monotone in each argument, so the
    inverse below is closed form. Avoiding a numerical inverse matters: the
    scalar curvature of the pushed-forward metric is a second derivative, and
    differentiating through a Newton solve is both slow and ill-conditioned.
    """
    return torch.stack([1.4 * z[0] + 0.2,
                        (0.8 + 0.1 * torch.tanh(z[0])) * z[1] - 0.3 * torch.sin(z[0])])


def diffeo_inv(w):
    z0 = (w[0] - 0.2) / 1.4
    z1 = (w[1] + 0.3 * torch.sin(z0)) / (0.8 + 0.1 * torch.tanh(z0))
    return torch.stack([z0, z1])


def invariants_analytic(drift_fn, sig_fn, z, jitter=1e-9):
    V = sig_fn(z) @ sig_fn(z).T
    def g(x):
        S = sig_fn(x)
        return torch.linalg.inv(S @ S.T + jitter * torch.eye(2, dtype=x.dtype))
    a = drift_fn(z)
    Pe = a @ g(z) @ a
    R = scalar_curvature(g, z)
    return torch.stack([R, Pe])


def part_a():
    print('A. analytic push-forward invariance')
    torch.manual_seed(0)
    pts = torch.stack([torch.rand(40) * 3 - 1.5, torch.rand(40) * 2 - 1.0], 1)

    # the pushed-forward system, expressed as functions of the new coordinate w
    def drift_w(w):
        z = diffeo_inv(w)
        return jacrev(diffeo)(z) @ drift_strat_z(z)

    def sig_w(w):
        z = diffeo_inv(w)
        return jacrev(diffeo)(z) @ sig_z(z)

    I0 = torch.stack([invariants_analytic(drift_strat_z, sig_z, p) for p in pts])
    I1 = torch.stack([invariants_analytic(drift_w, sig_w, diffeo(p)) for p in pts])
    err = (I1 - I0).abs()
    rel = err / I0.abs().clamp_min(1e-9)
    print(f'   R  spans [{I0[:,0].min():+.3f}, {I0[:,0].max():+.3f}]   '
          f'max |dR|  = {err[:,0].max():.3e}  (rel {rel[:,0].max():.2e})')
    print(f'   Pe spans [{I0[:,1].min():+.3f}, {I0[:,1].max():+.3f}]   '
          f'max |dPe| = {err[:,1].max():.3e}  (rel {rel[:,1].max():.2e})')
    return rel.max().item()


# ----------------------------------------------------------------------------
def simulate(n_traj, n_steps, dt, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.stack([torch.rand(n_traj, generator=g) * 3 - 1.5,
                     torch.randn(n_traj, generator=g) * 0.6], -1)
    Z = []
    bs = vmap(drift_strat_z)
    ss = vmap(sig_z)
    for _ in range(n_steps):
        Z.append(z.clone())
        dw = torch.randn(n_traj, 2, generator=g) * dt ** 0.5
        # Stratonovich -> Ito correction for simulation via Heun (midpoint)
        pred = z + bs(z) * dt + torch.einsum('nij,nj->ni', ss(z), dw)
        z = z + 0.5 * (bs(z) + bs(pred)) * dt \
            + 0.5 * torch.einsum('nij,nj->ni', ss(z) + ss(pred), dw)
    return torch.stack(Z, 1)          # [n_traj, n_steps, 2]


def fit(Z, DZ, dt, k, epochs=120, width=96, seed=0, lr=3e-3):
    torch.manual_seed(seed)
    sde = LatentSDE(k=k, width=width, depth=3)
    opt = torch.optim.Adam(sde.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, bs = Z.shape[0], min(8192, Z.shape[0])
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = sde.nll(Z[idx], DZ[idx], dt)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return sde


def spear(a, b):
    ra, rb = [np.argsort(np.argsort(x.numpy())) for x in (a, b)]
    return float(np.corrcoef(ra, rb)[0, 1])


def part_b(dt=0.01, n_traj=150, n_steps=400):
    print('\nB. invariance after estimation from finite samples')
    traj = simulate(n_traj, n_steps, dt, seed=1)
    Z = traj[:, :-1].reshape(-1, 2)
    DZ = (traj[:, 1:] - traj[:, :-1]).reshape(-1, 2)
    W = vmap(diffeo)(Z)
    Wn = vmap(diffeo)(traj[:, 1:].reshape(-1, 2))
    DW = Wn - W
    print(f'   {Z.shape[0]} transitions, dt={dt}')

    grid = torch.stack([torch.rand(300) * 2.4 - 1.2, torch.rand(300) * 1.4 - 0.7], 1)
    gridw = vmap(diffeo)(grid)
    I_true = torch.stack([invariants_analytic(drift_strat_z, sig_z, p) for p in grid])

    rows = []
    for seed in range(3):
        sz = fit(Z, DZ, dt, 2, seed=seed)
        sw = fit(W, DW, dt, 2, seed=seed)
        Iz = Invariants(sz)(grid)
        Iw = Invariants(sw)(gridw)
        rows.append(dict(
            R_zw=spear(Iz[:, 0], Iw[:, 0]), Pe_zw=spear(Iz[:, 1], Iw[:, 1]),
            R_true=spear(Iz[:, 0], I_true[:, 0]), Pe_true=spear(Iz[:, 1], I_true[:, 1]),
            Pe_relerr=float(((Iz[:, 1] - Iw[:, 1]).abs()
                             / Iz[:, 1].abs().clamp_min(1e-6)).median()),
        ))
    for key in rows[0]:
        v = [r[key] for r in rows]
        print(f'   {key:10s} {np.mean(v):+.3f} +- {np.std(v):.3f}')

    # the non-invariant control: raw Tr(V) in the two coordinate systems
    with torch.no_grad():
        trz = torch.diagonal(sz.V(grid), dim1=-2, dim2=-1).sum(-1)
        trw = torch.diagonal(sw.V(gridw), dim1=-2, dim2=-1).sum(-1)
    print(f'   {"TrV_zw":10s} {spear(trz, trw):+.3f}   <- basis dependent control')
    return rows


def part_c():
    print('\nC. cost of the invariants against latent dimension')
    for k in [2, 3, 4, 5, 6, 8]:
        sde = LatentSDE(k=k, width=96, depth=3)
        inv = Invariants(sde)
        Z = torch.randn(64, k)
        inv(Z[:8])                                  # warm up
        t0 = time.time(); inv(Z); dt = (time.time() - t0) / 64
        print(f'   k={k}: {1000*dt:8.2f} ms/point   ({1/dt:8.0f} points/s)')


if __name__ == '__main__':
    e = part_a()
    part_b()
    print('\nanalytic invariance ->', 'PASS' if e < 1e-6 else f'FAIL ({e:.2e})')
