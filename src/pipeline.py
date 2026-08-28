"""
The shared IOO pipeline: chart -> latent SDE -> invariants -> amortised head.

Computing (R, Pe) exactly costs a few milliseconds per point, which is fine for
a design study and far too slow for 1.3M telemetry samples or for a real-time
controller. The invariants are therefore computed exactly on a subsample and
distilled into a small feed-forward head that maps chart coordinates directly
to (R, Pe). The head is what runs at inference; its fidelity against the exact
invariants is reported rather than assumed.
"""

import numpy as np
import torch
import torch.nn as nn

from manifold import Chart, LatentSDE, Invariants, mlp


def fit_chart(X, k, epochs=60, width=128, bs=4096, lr=2e-3, seed=0, verbose=False):
    """Autoencoder chart on standardised ambient states X [N,d]."""
    torch.manual_seed(seed)
    ch = Chart(X.shape[1], k, width=width).to(X.dtype)
    opt = torch.optim.Adam(ch.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xr, _ = ch(X[idx])
            loss = ((xr - X[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sch.step()
        if verbose and ep % 20 == 0:
            print(f'    chart ep {ep:3d}  mse {tot/n:.5f}')
    return ch


def fit_sde(Z, DZ, dt, epochs=60, width=128, bs=8192, lr=2e-3, seed=0, verbose=False):
    torch.manual_seed(seed)
    sde = LatentSDE(k=Z.shape[1], width=width, depth=3).to(Z.dtype)
    opt = torch.optim.Adam(sde.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = Z.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = sde.nll(Z[idx], DZ[idx], dt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sch.step()
        if verbose and ep % 20 == 0:
            print(f'    sde   ep {ep:3d}  nll  {tot/n:+.4f}')
    return sde


class InvariantHead(nn.Module):
    """Amortised (R, Pe): chart coordinate -> invariants, in microseconds."""

    def __init__(self, k, width=128):
        super().__init__()
        self.f = mlp([k, width, width, 2])
        self.register_buffer('mu', torch.zeros(2))
        self.register_buffer('sd', torch.ones(2))

    def forward(self, z):
        return self.f(z) * self.sd + self.mu


def distill_invariants(sde, Z_pool, n_exact=20000, epochs=300, seed=0,
                       chunk=256, verbose=True):
    """Compute exact invariants on a subsample, fit the head, report fidelity.

    Pe is heavy tailed, so the head is trained on signed-log targets; the
    reported R^2 is on that scale and a rank correlation is given alongside,
    because what the router consumes is essentially the ordering.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(Z_pool.shape[0], generator=g)[:n_exact]
    Zs = Z_pool[idx]
    inv = Invariants(sde)
    F = inv(Zs, chunk=chunk)                                   # [n,2] exact

    def slog(t):
        return torch.sign(t) * torch.log1p(t.abs())

    T = slog(F)
    ntr = int(0.8 * len(T))
    head = InvariantHead(sde.k).to(Zs.dtype)
    head.mu.copy_(T[:ntr].mean(0)); head.sd.copy_(T[:ntr].std(0) + 1e-8)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = torch.randperm(ntr)
        for i in range(0, ntr, 2048):
            j = perm[i:i + 2048]
            loss = ((head(Zs[j]) - T[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

    with torch.no_grad():
        pred = head(Zs[ntr:])
    tgt = T[ntr:]
    ss_res = ((pred - tgt) ** 2).sum(0)
    ss_tot = ((tgt - tgt.mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).tolist()
    rho = []
    for c in range(2):
        a = np.argsort(np.argsort(pred[:, c].numpy()))
        b = np.argsort(np.argsort(tgt[:, c].numpy()))
        rho.append(float(np.corrcoef(a, b)[0, 1]))
    if verbose:
        print(f'    invariant head: R^2 (R,Pe) = {r2[0]:.3f}, {r2[1]:.3f} | '
              f'spearman = {rho[0]:.3f}, {rho[1]:.3f}')
    return head, dict(r2_R=r2[0], r2_Pe=r2[1], rho_R=rho[0], rho_Pe=rho[1],
                      n_exact=int(n_exact)), F, Zs


@torch.no_grad()
def apply_head(head, Z, bs=200000):
    return torch.cat([head(Z[i:i + bs]) for i in range(0, Z.shape[0], bs)])


class IOOFrontEnd:
    """Chart + SDE + invariant head, fitted without labels.

    Unsupervised throughout: adapting the front end to a new fleet, a new
    calibration, or a new operating envelope needs telemetry only.
    """

    def __init__(self, k=3, dt=0.5, seed=0):
        self.k, self.dt, self.seed = k, dt, seed

    def fit(self, S, G, chart_epochs=60, sde_epochs=60, n_exact=20000, verbose=True):
        # The front end runs in float64 throughout: the scalar curvature is a
        # second derivative of the diffusion network, and in float32 the
        # cancellation in the Christoffel differences dominates the signal.
        S = S.double()
        self.mu_ = S.mean(0); self.sd_ = S.std(0) + 1e-8
        Xs = (S - self.mu_) / self.sd_
        self.chart = fit_chart(Xs, self.k, epochs=chart_epochs, seed=self.seed,
                               verbose=verbose)
        with torch.no_grad():
            Z = self.chart.enc(Xs)
        self.z_mu_, self.z_sd_ = Z.mean(0), Z.std(0) + 1e-8
        Z = (Z - self.z_mu_) / self.z_sd_
        same = G[:-1] == G[1:]
        self.sde = fit_sde(Z[:-1][same], (Z[1:] - Z[:-1])[same], self.dt,
                           epochs=sde_epochs, seed=self.seed, verbose=verbose)
        self.head, self.fidelity, self._F, self._Zs = distill_invariants(
            self.sde, Z, n_exact=n_exact, seed=self.seed, verbose=verbose)
        return self

    @torch.no_grad()
    def encode(self, S):
        Z = self.chart.enc((S.double() - self.mu_) / self.sd_)
        return (Z - self.z_mu_) / self.z_sd_

    @torch.no_grad()
    def invariants(self, S, dtype=torch.float32):
        return apply_head(self.head, self.encode(S)).to(dtype)

    @torch.no_grad()
    def naive_features(self, S, dtype=torch.float32):
        """Tr V and log det V: the basis-dependent control."""
        Z = self.encode(S)
        out = []
        for i in range(0, Z.shape[0], 100000):
            V = self.sde.V(Z[i:i + 100000])
            tr = torch.diagonal(V, dim1=-2, dim2=-1).sum(-1)
            ld = torch.logdet(V + 1e-6 * torch.eye(self.k, dtype=Z.dtype))
            out.append(torch.stack([tr, ld], -1))
        return torch.cat(out).to(dtype)
