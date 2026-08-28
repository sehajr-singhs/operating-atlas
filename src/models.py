"""
Operator assemblies and the routing arms compared in the paper.

Every arm shares one expert bank and one training budget. The experts always
see the same inputs; only the router's input coordinates differ. That is the
whole controlled comparison: it isolates *what the routing decision is made on*
from every other design choice.

Arms
  mono        one network, width scaled to match the assembly's parameter count
  raw         router sees the raw (standardised) state
  invariant   router sees (R, Pe) only -- two coordinate invariants
  raw+inv     router sees both
  naive       router sees (Tr V, log det V) -- the basis-dependent stochastic
              features, the control for "is invariance what matters?"
  activity    router sees local speed and local variance -- the control for
              "does anything cheap and local do just as well?"
  random      router sees a fixed random 2-d projection of the state -- the
              control for "is it just extra router inputs?"
  oracle      router sees the ground-truth regime label, where one exists
"""

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, act=nn.SiLU, dropout=0.0):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(act())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------
# expert primitives
# ----------------------------------------------------------------------------

class AffineExpert(nn.Module):
    """The low-curvature, advection-dominated primitive: an affine map."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.f = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.f(x)


class MLPExpert(nn.Module):
    def __init__(self, d_in, d_out, width, depth=2, dropout=0.0):
        super().__init__()
        self.f = mlp([d_in] + [width] * depth + [d_out], dropout=dropout)

    def forward(self, x):
        return self.f(x)


class LowRankQuadExpert(nn.Module):
    """A curvature primitive: an explicit second-order form in a low-rank
    subspace, which is what a locally quadratic chart of the manifold needs."""

    def __init__(self, d_in, d_out, rank=8):
        super().__init__()
        self.P = nn.Linear(d_in, rank, bias=False)
        self.lin = nn.Linear(d_in, d_out)
        self.quad = nn.Linear(rank * (rank + 1) // 2, d_out, bias=False)
        idx = torch.triu_indices(rank, rank)
        self.register_buffer('ii', idx[0])
        self.register_buffer('jj', idx[1])

    def forward(self, x):
        u = self.P(x)
        q = u[:, self.ii] * u[:, self.jj]
        return self.lin(x) + self.quad(q)


def build_experts(kind, d_in, d_out, K, width, dropout=0.0):
    if kind == 'mlp':
        return nn.ModuleList([MLPExpert(d_in, d_out, width, dropout=dropout) for _ in range(K)])
    if kind == 'physics':
        # heterogeneous primitives, cycled: affine / quadratic / nonlinear
        ctor = [lambda: AffineExpert(d_in, d_out),
                lambda: LowRankQuadExpert(d_in, d_out, rank=min(12, d_in)),
                lambda: MLPExpert(d_in, d_out, width, dropout=dropout)]
        return nn.ModuleList([ctor[i % 3]() for i in range(K)])
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------

class OperatorAssembly(nn.Module):
    """y(x) = sum_k p_k(r) O_k(x),  p = softmax(u(r)/tau).

    r is the routing coordinate and is supplied separately from x.
    """

    def __init__(self, d_in, d_out, d_route, K=6, width=64, router_width=64,
                 tau=1.0, expert='mlp', dropout=0.0):
        super().__init__()
        self.K, self.tau = K, tau
        self.experts = build_experts(expert, d_in, d_out, K, width, dropout)
        self.router = mlp([d_route, router_width, router_width, K])

    def gate(self, r):
        return torch.softmax(self.router(r) / self.tau, -1)

    def forward(self, x, r, return_gate=False):
        p = self.gate(r)
        ys = torch.stack([e(x) for e in self.experts], 1)       # [N,K,d_out]
        y = (p.unsqueeze(-1) * ys).sum(1)
        return (y, p, ys) if return_gate else y


class Monolith(nn.Module):
    def __init__(self, d_in, d_out, width, depth=4, dropout=0.0):
        super().__init__()
        self.f = mlp([d_in] + [width] * depth + [d_out], dropout=dropout)

    def forward(self, x, r=None):
        return self.f(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def match_monolith_width(d_in, d_out, target, depth=4, lo=8, hi=2048):
    """Smallest width whose monolith parameter count first meets `target`."""
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(Monolith(d_in, d_out, mid, depth))
        if n < target:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return best + 1


# ----------------------------------------------------------------------------
# Theorem 1: exact sensitivity identity
# ----------------------------------------------------------------------------

def gate_sensitivity_identity(model: OperatorAssembly, x, r):
    """Verify  dF/dr = (1/tau) Cov_{k~p}( du_k/dr , O_k(x) )  to machine precision.

    The identity says the assembly's sensitivity to the routing coordinate is
    the gate-weighted covariance between router logit gradients and expert
    outputs. Transitions across operating regimes are therefore smooth exactly
    when the experts agree wherever the gate is moving quickly -- a design
    statement, not the vacuous observation that a softmax of smooth maps is
    smooth.

    Returns (max abs discrepancy, ||dF/dr||, disagreement, gate entropy).
    """
    x = x.detach(); r = r.detach().requires_grad_(True)
    p = model.gate(r)                                     # [N,K]
    ys = torch.stack([e(x) for e in model.experts], 1)    # [N,K,do]
    y = (p.unsqueeze(-1) * ys).sum(1)

    # autodiff dF/dr for the first output channel
    gy = torch.autograd.grad(y[:, 0].sum(), r, create_graph=False, retain_graph=True)[0]

    u = model.router(r)
    du = torch.stack([torch.autograd.grad(u[:, k].sum(), r, retain_graph=True)[0]
                      for k in range(model.K)], 1)        # [N,K,dr]
    o = ys[:, :, 0]                                       # [N,K]
    Ebar = (p.unsqueeze(-1) * du).sum(1, keepdim=True)    # [N,1,dr]
    obar = (p * o).sum(1, keepdim=True)                   # [N,1]
    cov = (p.unsqueeze(-1) * (du - Ebar) * (o - obar).unsqueeze(-1)).sum(1)
    pred = cov / model.tau

    disagreement = ((p * (o - obar) ** 2).sum(1)).sqrt()
    ent = -(p * (p + 1e-12).log()).sum(1)
    return ((gy - pred).abs().max().item(), gy.norm(dim=-1).detach(),
            disagreement.detach(), ent.detach())


# ----------------------------------------------------------------------------
# routing coordinate builders
# ----------------------------------------------------------------------------

def random_projection(X, dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(X.shape[1], dim, generator=g, dtype=X.dtype) / np.sqrt(X.shape[1])
    return X @ P
