"""
Is the diffusion assumption actually satisfied?

The whole construction rests on eq. (1): increments are locally Gaussian with
covariance V(z) dt. That is a precondition, not a modelling preference, and it
is directly checkable without reference to any downstream accuracy.

For each dataset we take the fitted SDE, whiten every held-out increment by its
own predicted Cholesky factor,

    e = L(z)^{-1} (dz - b(z) dt) / sqrt(dt),

and ask whether e looks standard normal. Under a correctly specified diffusion
it does. Under a jump process it does not: whitened increments acquire heavy
tails, because the fitted V has to average over both within-regime noise and
between-regime jumps, so ordinary steps are over-shrunk and jumps remain huge.

Reported per dataset: excess kurtosis of the whitened increments (0 for a
Gaussian), the fraction beyond 5 sigma (2.9e-7 for a Gaussian), and the
Kolmogorov-Smirnov distance of the whitened radial statistic against its
chi distribution. These are descriptive: with 1e5+ samples any normality test
rejects, so effect size is what matters, not a p-value.
"""

import json
import os
import sys

import numpy as np
import torch
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def whiten(front, S, G, dt, max_n=400000):
    Z = front.encode(S)
    same = (G[:-1] == G[1:]).numpy()
    z, dz = Z[:-1][same], (Z[1:] - Z[:-1])[same]
    if len(z) > max_n:
        idx = np.random.RandomState(0).choice(len(z), max_n, replace=False)
        z, dz = z[idx], dz[idx]
    out = []
    with torch.no_grad():
        for i in range(0, len(z), 100000):
            zz, dd = z[i:i + 100000], dz[i:i + 100000]
            L = front.sde.L(zz) * dt ** 0.5
            r = (dd - front.sde.b(zz) * dt).unsqueeze(-1)
            out.append(torch.linalg.solve_triangular(L, r, upper=False).squeeze(-1))
    return torch.cat(out).numpy()


def report(name, e):
    k = e.shape[1]
    flat = e.ravel()
    kurt = float(stats.kurtosis(flat))
    tail5 = float((np.abs(flat) > 5).mean())
    rad = np.linalg.norm(e, axis=1)
    ks = float(stats.kstest(rad, lambda x: stats.chi.cdf(x, df=k)).statistic)
    row = dict(dataset=name, k=k, n=int(len(e)), excess_kurtosis=kurt,
               frac_beyond_5sigma=tail5, gaussian_frac_5sigma=5.73e-7,
               tail_excess_ratio=tail5 / 5.73e-7 if tail5 > 0 else 0.0,
               ks_radial=ks, sd=float(flat.std()))
    print(f'  {name:10s} k={k}  n={len(e):7d}  excess kurtosis {kurt:9.2f}  '
          f'>5sigma {tail5:.2e} ({row["tail_excess_ratio"]:8.0f}x Gaussian)  '
          f'KS(radial) {ks:.3f}')
    return row


def main():
    names = sys.argv[1:] or ['pmsm', 'cmapss', 'ur5e']
    dts = {'pmsm': 0.5 * 4, 'cmapss': 1.0, 'ur5e': 0.02 * 2,
           'panda': 0.02 * 2, 'iiwa14': 0.02 * 2}
    rows = []
    print('Whitened-increment diagnostics (Gaussian => kurtosis 0, KS ~ 0):')
    for n in names:
        f = os.path.join(DATA, f'prep_{n}_k3_front.pt')
        b = os.path.join(DATA, f'prep_{n}_k3.pt')
        if not (os.path.exists(f) and os.path.exists(b)):
            print(f'  {n:10s} not prepped, skipping')
            continue
        front = torch.load(f, weights_only=False)
        blob = torch.load(b, weights_only=False)
        e = whiten(front, blob['S']['test'], blob['G']['test'], dts.get(n, 1.0))
        rows.append(report(n, e))
    with open(os.path.join(RES, 'diffusion_check.json'), 'w') as fh:
        json.dump(rows, fh, indent=2)
    print('wrote results/diffusion_check.json')


if __name__ == '__main__':
    main()
