"""
Every PMSM number the paper quotes, computed once from the cached sessions and
written to results_pmsm.json, plus the results figure.

Runs off _pmsm_sessions.npz so it is seconds rather than minutes, and so the
figure and the text can never drift apart: whatever is plotted is whatever was
written to the json in the same pass.
"""

import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import atoms                                    # noqa: E402
from scipy.stats import skew, kurtosis          # noqa: E402

CACHE = os.path.join(HERE, '_pmsm_sessions.npz')
NS = [125, 250, 500, 1000, 2000, 4000, 8000]
NA = len(atoms.ATOM_NAMES)


def ident(S):
    n = len(S)
    r = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    return float((r == 0).mean() * 100), float((1 - r / (n - 1)).mean() * 100)


def cos_sim(A, B):
    A = np.nan_to_num(A); B = np.nan_to_num(B)
    keep = (A.std(0) > 1e-9) & (B.std(0) > 1e-9)
    if keep.sum() < 2:
        return None
    A, B = A[:, keep], B[:, keep]
    pool = np.concatenate([A, B]); mu, sd = pool.mean(0), pool.std(0) + 1e-9
    A, B = (A - mu) / sd, (B - mu) / sd
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def atom_corr(A, B):
    out = {}
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        x, y = A[..., ai].ravel(), B[..., ai].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        out[nm] = (float(np.corrcoef(x[m], y[m])[0, 1])
                   if m.sum() > 10 and x[m].std() > 1e-9 and y[m].std() > 1e-9
                   else float('nan'))
    return out


def marginal(X):
    return np.concatenate([X.mean(0), X.std(0), skew(X, 0), kurtosis(X, 0)])


def main():
    t0 = time.time()
    z = np.load(CACHE)
    X_all = z['X']
    u, N, d = X_all.shape
    R = {'n_units': int(u), 'n_samples': int(N), 'n_channels': int(d),
         'atoms': atoms.ATOM_NAMES}
    print(f'{u} sessions x {N} x {d}', flush=True)

    # ---- 1. telemetry budget --------------------------------------------
    budget = {'N': NS, 'disjoint': {}, 'interleaved': {}}
    for kind in ['disjoint', 'interleaved']:
        acc = {nm: [] for nm in atoms.ATOM_NAMES}
        for n in NS:
            cs = {nm: [] for nm in atoms.ATOM_NAMES}
            for i in range(u):
                X = X_all[i].astype(np.float64)
                if kind == 'disjoint':
                    Xa, Xb = X[:n], X[-n:]
                else:
                    W = X[:2 * n]
                    Xa, Xb = W[0::2], W[1::2]
                A = atoms.atlas_unit(Xa, n_cells=1, min_cell=50)
                B = atoms.atlas_unit(Xb, n_cells=1, min_cell=50)
                for nm, v in atom_corr(A, B).items():
                    if np.isfinite(v):
                        cs[nm].append(v)
            for nm in atoms.ATOM_NAMES:
                acc[nm].append(float(np.mean(cs[nm])) if cs[nm] else float('nan'))
            print(f'  budget {kind} N={n} [{time.time()-t0:.0f}s]', flush=True)
        budget[kind] = acc
    R['budget'] = budget

    # ---- 2. identification ----------------------------------------------
    A, B, Am, Bm, Bw = [], [], [], [], []
    for i in range(u):
        X = X_all[i].astype(np.float64)
        Xa, Xb = X[:N // 2], X[N // 2:]
        Xw = atoms.warp_channels(Xb, np.random.default_rng(9000 + i))
        A.append(atoms.atlas_unit(Xa, n_cells=1, min_cell=50))
        B.append(atoms.atlas_unit(Xb, n_cells=1, min_cell=50))
        Bw.append(atoms.atlas_unit(Xw, n_cells=1, min_cell=50))
        Am.append(marginal(Xa)); Bm.append(marginal(Xb))
    A, B, Bw = np.array(A), np.array(B), np.array(Bw)
    Am, Bm = np.nan_to_num(np.array(Am)), np.nan_to_num(np.array(Bm))
    Bmw = np.nan_to_num(np.array(
        [marginal(atoms.warp_channels(X_all[i, N // 2:].astype(np.float64),
                                      np.random.default_rng(9000 + i)))
         for i in range(u)]))

    inv = [i for i, nm in enumerate(atoms.ATOM_NAMES) if nm in atoms.WARP_INVARIANT]
    fl = lambda M: M.reshape(len(M), -1)
    sets = {
        'full atlas (9)':      (fl(A), fl(B)),
        'atlas, invariant (8)': (fl(A[..., inv]), fl(B[..., inv])),
        'rho only':            (fl(A[..., 0:1]), fl(B[..., 0:1])),
        'marginals':           (Am, Bm),
    }
    R['identification'] = {}
    for k, (a, b) in sets.items():
        s = cos_sim(a, b)
        t1, pr = ident(s)
        R['identification'][k] = dict(top1=t1, pct=pr)
        print(f'  ident {k:<22} top1={t1:5.1f}%  pct={pr:5.1f}%', flush=True)
    R['identification']['chance'] = dict(top1=100.0 / u, pct=50.0)

    R['per_atom'] = {}
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        s = cos_sim(A[..., ai], B[..., ai])
        t1, pr = ident(s)
        R['per_atom'][nm] = dict(top1=t1, pct=pr)

    # ---- 3. invariance ---------------------------------------------------
    R['invariance'] = {
        'atom_shift': {nm: float(np.nanmax(np.abs(B[..., ai] - Bw[..., ai])))
                       for ai, nm in enumerate(atoms.ATOM_NAMES)},
    }
    s = cos_sim(fl(A[..., inv]), fl(Bw[..., inv])); t1, pr = ident(s)
    R['invariance']['atlas_warped'] = dict(top1=t1, pct=pr)
    s = cos_sim(Am, Bmw); t1w, prw = ident(s)
    R['invariance']['marginal_warped'] = dict(top1=t1w, pct=prw)
    print(f'  warped: atlas top1={t1:.1f}%  marginal top1={t1w:.1f}%', flush=True)

    with open(os.path.join(HERE, 'results_pmsm.json'), 'w') as f:
        json.dump(R, f, indent=2)
    print(f'wrote results_pmsm.json [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
