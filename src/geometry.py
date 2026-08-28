"""
Riemannian geometry primitives for the IOO framework.

Everything here operates on a metric supplied as a callable g: R^d -> S^d_{++}.
Christoffel symbols, the Riemann tensor, Ricci and the scalar curvature R(s)
are obtained by automatic differentiation. Correctness is pinned against
closed-form curvatures in test_geometry().

Index convention (0-based, all indices lower-case latin):
    g[i,j]        metric
    Gamma[k,i,j]  Christoffel symbol of the second kind, Gamma^k_{ij}
    Riem[l,i,j,k] R^l_{ijk}
    Ric[j,k]      R^i_{ijk}
    R             g^{jk} Ric_{jk}
"""

import torch
from torch.func import jacrev, vmap


# ----------------------------------------------------------------------------
# core tensors
# ----------------------------------------------------------------------------

def christoffel(g_fn, s):
    """Gamma^k_{ij} at a single point s of shape [d]. Returns [d,d,d].

    jacrev(g_fn)(s) has layout A[b,c,a] = d g_bc / d s^a, so
        d_i g_lj = A[l,j,i],  d_j g_li = A[l,i,j],  d_l g_ij = A[i,j,l].
    """
    ginv = torch.linalg.inv(g_fn(s))
    A = jacrev(g_fn)(s)
    return 0.5 * (torch.einsum('kl,lji->kij', ginv, A)
                  + torch.einsum('kl,lij->kij', ginv, A)
                  - torch.einsum('kl,ijl->kij', ginv, A))


def riemann(g_fn, s):
    """R^l_{ijk} at s. Returns [d,d,d,d]."""
    Gam = christoffel(g_fn, s)                                # [k,i,j]
    dGam = jacrev(lambda x: christoffel(g_fn, x))(s)          # [k,i,j,m] = d Gam^k_ij / d s^m

    # R^l_{ijk} = d_i Gam^l_{jk} - d_j Gam^l_{ik}
    #             + Gam^l_{im} Gam^m_{jk} - Gam^l_{jm} Gam^m_{ik}
    d_i_G_ljk = dGam.permute(0, 3, 1, 2)   # [l,i,j,k]
    d_j_G_lik = dGam.permute(0, 1, 3, 2)   # [l,i,j,k]
    t3 = torch.einsum('lim,mjk->lijk', Gam, Gam)
    t4 = torch.einsum('ljm,mik->lijk', Gam, Gam)
    return d_i_G_ljk - d_j_G_lik + t3 - t4


def ricci(g_fn, s):
    """Ric_{jk} = R^i_{ijk}. Returns [d,d]."""
    return torch.einsum('iijk->jk', riemann(g_fn, s))


def scalar_curvature(g_fn, s):
    """R = g^{jk} Ric_{jk}. A coordinate invariant. Returns a scalar."""
    Ric = torch.einsum('iijk->jk', riemann(g_fn, s))
    ginv = torch.linalg.inv(g_fn(s))
    return torch.einsum('jk,jk->', ginv, Ric)


def batch_scalar_curvature(g_fn, S, chunk=128):
    """Scalar curvature over a batch S [N,d]; chunked to bound memory."""
    f = lambda x: scalar_curvature(g_fn, x)
    return torch.cat([vmap(f)(S[i:i + chunk]) for i in range(0, S.shape[0], chunk)])


# ----------------------------------------------------------------------------
# metrics induced by learned dynamics
# ----------------------------------------------------------------------------

def pullback_metric_fn(f, eps=1e-3):
    """g(s) = J_f(s)^T J_f(s) + eps I, the pullback of the euclidean metric
    through f. With f the tau-ahead flow map this is the right Cauchy-Green
    strain tensor, whose spectrum gives finite-time Lyapunov exponents.
    eps regularises the inverse where the flow is locally rank-deficient."""
    def g(s):
        J = jacrev(f)(s)
        return J.transpose(-1, -2) @ J + eps * torch.eye(s.shape[0], dtype=s.dtype)
    return g


def ftle(f, s, tau=1.0):
    """Largest finite-time Lyapunov exponent of the flow map f over horizon tau."""
    sv = torch.linalg.svdvals(jacrev(f)(s))
    return torch.log(sv[0]) / tau


def invariant_diffusion_trace(g_fn, V_fn, s):
    """Tr(g^{-1} V), the metric contraction of the diffusion tensor.

    Plain Tr(V) is basis dependent and changes under a smooth relabelling of the
    state coordinates; the contraction against the inverse metric is a genuine
    scalar invariant, and is what the router should consume."""
    return torch.einsum('ij,ij->', torch.linalg.inv(g_fn(s)), V_fn(s))


# ----------------------------------------------------------------------------
# closed-form validation targets
# ----------------------------------------------------------------------------

def sphere_metric(r=1.0):
    """2-sphere of radius r in (theta, phi). Scalar curvature R = 2/r^2."""
    def g(s):
        z = torch.zeros((), dtype=s.dtype)
        rr = torch.tensor(r ** 2, dtype=s.dtype)
        return torch.stack([torch.stack([rr, z]),
                            torch.stack([z, rr * torch.sin(s[0]) ** 2])])
    return g, 2.0 / r ** 2


def hyperbolic_metric(k=1.0):
    """Scaled Poincare half-plane, g = (k/y^2) I. Scalar curvature R = -2/k."""
    def g(s):
        z = torch.zeros((), dtype=s.dtype)
        v = k / s[1] ** 2
        return torch.stack([torch.stack([v, z]), torch.stack([z, v])])
    return g, -2.0 / k


def torus_metric(R_maj=2.0, r_min=1.0):
    """Torus in (u,v): g = diag((R + r cos v)^2, r^2).
    Gaussian curvature K = cos v / (r (R + r cos v)), scalar curvature R = 2K."""
    def g(s):
        z = torch.zeros((), dtype=s.dtype)
        return torch.stack([
            torch.stack([(R_maj + r_min * torch.cos(s[1])) ** 2, z]),
            torch.stack([z, torch.tensor(r_min ** 2, dtype=s.dtype)])])

    def R_exact(s):
        return 2 * torch.cos(s[1]) / (r_min * (R_maj + r_min * torch.cos(s[1])))
    return g, R_exact


def product_sphere_metric(r=1.0):
    """S^2(r) x R^2 in 4 coordinates: checks that flat directions contribute
    nothing. R = 2/r^2."""
    def g(s):
        z = torch.zeros((), dtype=s.dtype)
        one = torch.ones((), dtype=s.dtype)
        rr = torch.tensor(r ** 2, dtype=s.dtype)
        return torch.stack([
            torch.stack([rr, z, z, z]),
            torch.stack([z, rr * torch.sin(s[0]) ** 2, z, z]),
            torch.stack([z, z, one, z]),
            torch.stack([z, z, z, one])])
    return g, 2.0 / r ** 2


def test_geometry(verbose=True):
    """Pin the curvature code against closed forms. Returns max abs errors."""
    torch.manual_seed(0)
    errs = {}

    g, R_true = sphere_metric(r=1.7)
    pts = torch.stack([torch.rand(20) * 2.0 + 0.4, torch.rand(20) * 6.0], 1).double()
    errs['sphere r=1.7'] = (batch_scalar_curvature(g, pts) - R_true).abs().max().item()

    g, R_true = hyperbolic_metric(k=2.3)
    pts = torch.stack([torch.rand(20) * 4 - 2, torch.rand(20) * 2 + 0.5], 1).double()
    errs['hyperbolic k=2.3'] = (batch_scalar_curvature(g, pts) - R_true).abs().max().item()

    g, R_fn = torus_metric(2.0, 1.0)
    pts = torch.stack([torch.rand(20) * 6, torch.rand(20) * 6], 1).double()
    Rt = torch.stack([R_fn(p) for p in pts])
    errs['torus'] = (batch_scalar_curvature(g, pts) - Rt).abs().max().item()

    g, R_true = product_sphere_metric(r=1.2)
    pts = torch.cat([torch.rand(20, 1) * 2 + 0.4, torch.rand(20, 3) * 4 - 2], 1).double()
    errs['S2(1.2) x R2'] = (batch_scalar_curvature(g, pts) - R_true).abs().max().item()

    gflat = lambda s: torch.eye(5, dtype=s.dtype)
    errs['flat R5'] = batch_scalar_curvature(gflat, torch.randn(10, 5).double()).abs().max().item()

    # Scalar curvature must be unchanged by a diffeomorphic change of coordinates.
    gs, R_true = sphere_metric(r=1.0)

    def phi(u):
        return torch.stack([u[0] + 0.3 * torch.sin(u[1]), u[1] + 0.2 * u[0]])

    def g_pulled(u):
        J = jacrev(phi)(u)
        return J.T @ gs(phi(u)) @ J

    pts = torch.stack([torch.rand(15) * 1.2 + 0.6, torch.rand(15) * 2], 1).double()
    errs['reparam invariance'] = (batch_scalar_curvature(g_pulled, pts) - R_true).abs().max().item()

    if verbose:
        for k, v in errs.items():
            print(f'  {k:22s} max|R_hat - R_true| = {v:.3e}')
    return errs


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    print('Validating curvature implementation against closed forms:')
    errs = test_geometry()
    worst = max(errs.values())
    print(f'\nworst error {worst:.3e} ->', 'PASS' if worst < 1e-6 else 'FAIL')
