"""
Chart maps, neural Kramers-Moyal estimation, and the invariant routing
coordinates of the IOO framework.

Three objects are estimated from telemetry alone:

  1. a chart  phi: R^d -> R^k  taking the ambient sensor vector to local
     manifold coordinates (an autoencoder; the decoder is the embedding),
  2. the Ito drift b(z) and diffusion V(z) = Sigma Sigma^T of the latent SDE,
     estimated as conditional Kramers-Moyal moments,
  3. two coordinate invariants built from them.

The metric. For a non-degenerate diffusion the canonical Riemannian structure
on the state space is

        g(z) = V(z)^{-1},

the metric in which Varadhan's short-time heat-kernel asymptotics reduce to the
geodesic distance, -4t log p(z,z',t) -> d_g(z,z')^2. The volatility field is
therefore not a separate quantity to be concatenated alongside the geometry: it
*is* the geometry. Everything the router consumes is a scalar built from g.

The invariants.
        R(z)    scalar curvature of g -- how the volatility geometry bends
        Pe(z)   = ||a||_g^2 = a^T V^{-1} a, with a the Stratonovich drift.
                A squared drift-to-noise ratio: a local Peclet number saying
                whether transport at this operating point is advective
                (deterministic, Pe >> 1) or diffusive (noise dominated, Pe << 1).

Both are unchanged by any smooth invertible relabelling of the state
coordinates. Under z' = f(z) the Stratonovich calculus obeys the classical
chain rule, so a' = J a and V' = J V J^T, whence
        Pe' = a^T J^T J^{-T} V^{-1} J^{-1} J a = Pe.
This is why the Stratonovich convention matters: under Ito the drift acquires a
correction term under change of variables and Pe would only be approximately
invariant. The fitted Euler-Maruyama drift is an Ito drift, so it is converted.

Smooth activations are mandatory: R is a second derivative of the network that
carries the metric, so ReLU-type nets give an identically zero curvature field.
"""

import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from geometry import scalar_curvature, batch_scalar_curvature


def mlp(sizes, act=nn.Tanh):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])


class Chart(nn.Module):
    """Autoencoder whose encoder is the chart and decoder the embedding."""

    def __init__(self, d, k, width=128, depth=3):
        super().__init__()
        self.d, self.k = d, k
        self.enc = mlp([d] + [width] * depth + [k])
        self.dec = mlp([k] + [width] * depth + [d])

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class LatentSDE(nn.Module):
    """Conditional first and second Kramers-Moyal moments of the latent process.

        b(z)            = E[dz | z] / dt                   (Ito drift)
        V(z) = L L^T    = Cov[dz | z] / dt

    Fitted by exact heteroscedastic Gaussian NLL on one-step transitions, the
    maximum-likelihood form of the Euler-Maruyama transition density.
    """

    def __init__(self, k, width=128, depth=3, min_sd=1e-3):
        super().__init__()
        self.k, self.min_sd = k, min_sd
        n = k * (k + 1) // 2
        self.b = mlp([k] + [width] * depth + [k])
        self.tril = mlp([k] + [width] * depth + [n])
        # Scatter the n free entries into a [k,k] lower triangle by a fixed
        # linear map rather than by advanced indexing: index_put is not
        # vmap-traceable, and every invariant is computed under vmap.
        ij = torch.tril_indices(k, k)
        E = torch.zeros(n, k * k)
        E[torch.arange(n), ij[0] * k + ij[1]] = 1.0
        self.register_buffer('E', E)
        self.register_buffer('diagm', torch.eye(k))
        self.register_buffer('offd', torch.tril(torch.ones(k, k), -1))

    def L(self, z):
        """Cholesky factor of V(z), shape [...,k,k]."""
        Lf = (self.tril(z) @ self.E).reshape(*z.shape[:-1], self.k, self.k)
        d = torch.nn.functional.softplus(Lf) + self.min_sd
        return Lf * self.offd + d * self.diagm

    def V(self, z):
        L = self.L(z)
        return L @ L.transpose(-1, -2)

    def nll(self, z, dz, dt):
        L = self.L(z) * dt ** 0.5
        r = (dz - self.b(z) * dt).unsqueeze(-1)
        sol = torch.linalg.solve_triangular(L, r, upper=False).squeeze(-1)
        logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        return (0.5 * (sol ** 2).sum(-1) + logdet).mean()

    # -- Ito -> Stratonovich ------------------------------------------------
    def strat_drift(self, z):
        """a^i = b^i - 1/2 sum_{j,k} Sigma^{jk} d_j Sigma^{ik}, single point."""
        dSig = jacrev(self.L)(z)                     # [i,k,j] = d Sigma^{ik} / d z^j
        Sig = self.L(z)
        corr = torch.einsum('jk,ikj->i', Sig, dSig)
        return self.b(z) - 0.5 * corr


class Invariants:
    """The coordinate-invariant routing coordinates (R, Pe).

    metric_mode:
      'diffusion'  g = V^{-1}                 (Varadhan; the default)
      'flow'       g = J_b^T J_b + eps I      (Cauchy-Green strain of the drift
                                               field; an ablation that ignores
                                               the noise geometry)
    """

    def __init__(self, sde: LatentSDE, metric_mode='diffusion', eps=1e-3, jitter=1e-6):
        self.sde, self.mode, self.eps, self.jitter = sde, metric_mode, eps, jitter
        self.k = sde.k

    def g(self, z):
        if self.mode == 'diffusion':
            V = self.sde.V(z)
            return torch.linalg.inv(V + self.jitter * torch.eye(self.k, dtype=z.dtype))
        J = jacrev(self.sde.b)(z)
        return J.T @ J + self.eps * torch.eye(self.k, dtype=z.dtype)

    def _point(self, z):
        g = self.g(z)
        a = self.sde.strat_drift(z)
        Pe = torch.einsum('i,ij,j->', a, g, a)
        R = scalar_curvature(self.g, z)
        return torch.stack([R, Pe])

    @torch.no_grad()
    def __call__(self, Z, chunk=256):
        return torch.cat([vmap(self._point)(Z[i:i + chunk])
                          for i in range(0, Z.shape[0], chunk)])

    @torch.no_grad()
    def curvature(self, Z, chunk=256):
        return batch_scalar_curvature(self.g, Z, chunk=chunk)


NAMES = ['R', 'Pe']


# ----------------------------------------------------------------------------
# non-invariant controls (used to show that invariance is what does the work)
# ----------------------------------------------------------------------------

class NaiveStochasticFeatures:
    """The features as originally proposed: raw Tr(V) and log det V. Both are
    basis dependent -- they change when the state coordinates are relabelled --
    and serve as the control against the invariant pair."""

    def __init__(self, sde: LatentSDE):
        self.sde = sde

    @torch.no_grad()
    def __call__(self, Z, chunk=4096):
        outs = []
        for i in range(0, Z.shape[0], chunk):
            V = self.sde.V(Z[i:i + chunk])
            tr = torch.diagonal(V, dim1=-2, dim2=-1).sum(-1)
            ld = torch.logdet(V + 1e-6 * torch.eye(self.sde.k, dtype=Z.dtype))
            outs.append(torch.stack([tr, ld], -1))
        return torch.cat(outs)


class ActivityFeatures:
    """A deliberately cheap non-geometric stand-in: local speed and local
    variance of the state. If this matches the invariant pair, the geometry is
    not earning its keep and the paper should say so."""

    @staticmethod
    def from_transitions(Z, DZ, k=32):
        from sklearn.neighbors import NearestNeighbors
        nn_ = NearestNeighbors(n_neighbors=k).fit(Z.numpy())
        _, idx = nn_.kneighbors(Z.numpy())
        speed = DZ.norm(dim=-1)
        loc_speed = speed[idx].mean(1)
        loc_var = DZ[idx].var(1).sum(-1)
        return torch.stack([loc_speed, loc_var], -1)
