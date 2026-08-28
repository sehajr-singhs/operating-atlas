"""
How much telemetry does an atlas actually need?

The C-MAPSS pilot returned a weak fingerprint (1.1 % top-1). There are two very
different explanations and they demand opposite responses:

  (a) there is no unit identity in the atlas -- the idea is wrong;
  (b) the atlas is a fine statistic that was estimated from ~100 samples across
      276 channel pairs, i.e. it was mostly noise -- the idea is untested.

This script separates them without appealing to any downstream task. For a
single unit, take two DISJOINT windows of N samples and compute an atlas on
each. If the atoms are estimable at that N, the two atlases agree. Sweeping N
gives a reproducibility curve, and the N at which it saturates is the telemetry
budget the method requires. That number is a deliverable in its own right: it
tells a practitioner whether their historian has enough data before they build
anything.

Run on the real PMSM bench, whose sessions are ~19,000 samples -- two orders of
magnitude longer than a C-MAPSS unit -- so the curve can actually be traced.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import atoms

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
PMSM_ALL = ['u_q', 'u_d', 'i_q', 'i_d', 'motor_speed', 'torque', 'ambient',
            'coolant', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
NS = [125, 250, 500, 1000, 2000, 4000, 8000]


def load_sessions(min_len=2 * max(NS)):
    df = pd.read_csv(os.path.join(ROOT, 'pmsm', 'measures_v2.csv'))
    out = []
    for pid, d in df.groupby('profile_id', sort=True):
        if len(d) < min_len:
            continue
        out.append((int(pid), d[PMSM_ALL].to_numpy(np.float64)))
    return out


def atom_corr(A, B):
    """Per-atom Pearson correlation between two atlas estimates, pooled over
    pairs. Correlation rather than error, because atoms live on different
    scales and what matters is whether the ORDERING of pairs is reproduced."""
    out = {}
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        x, y = A[:, ai], B[:, ai]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
            out[nm] = np.nan
        else:
            out[nm] = float(np.corrcoef(x[m], y[m])[0, 1])
    return out


if __name__ == '__main__':
    t0 = time.time()
    sess = load_sessions()
    print(f'{len(sess)} PMSM sessions with >= {2*max(NS)} samples '
          f'(lengths {min(len(s[1]) for s in sess)}-{max(len(s[1]) for s in sess)})',
          flush=True)

    rows = []
    for N in NS:
        accum = {nm: [] for nm in atoms.ATOM_NAMES}
        for pid, X in sess:
            if len(X) < 2 * N:
                continue
            # two disjoint contiguous windows, maximally separated, so the test
            # is not inflated by temporal autocorrelation between them
            A = atoms.atlas_unit(X[:N], n_cells=1, min_cell=50)[0]
            B = atoms.atlas_unit(X[-N:], n_cells=1, min_cell=50)[0]
            for nm, v in atom_corr(A, B).items():
                if np.isfinite(v):
                    accum[nm].append(v)
        rows.append((N, {nm: (np.mean(v) if v else np.nan) for nm, v in accum.items()}))
        print(f'  N={N:5d}  ' + '  '.join(
            f'{nm}={rows[-1][1][nm]:+.2f}' for nm in atoms.ATOM_NAMES)
            + f'   [{time.time()-t0:.0f}s]', flush=True)

    np.savez('pilot_repro.npz',
             NS=np.array([r[0] for r in rows]),
             names=np.array(atoms.ATOM_NAMES),
             corr=np.array([[r[1][nm] for nm in atoms.ATOM_NAMES] for r in rows]))
    print('\nsaved pilot_repro.npz')

    # ---- fingerprinting on the SAME long real sessions -------------------
    print('\n--- P2 fingerprint on PMSM sessions (long real records) ---',
          flush=True)
    N = 8000
    A, B, ids = [], [], []
    for pid, X in sess:
        if len(X) < 2 * N:
            continue
        A.append(atoms.atlas_unit(X[:N], n_cells=1, min_cell=50).reshape(-1))
        B.append(atoms.atlas_unit(X[-N:], n_cells=1, min_cell=50).reshape(-1))
        ids.append(pid)
    A, B = np.array(A), np.array(B)

    def fp(A, B, tag, k=None):
        A = np.nan_to_num(A); B = np.nan_to_num(B)
        keep = (A.std(0) > 1e-9) & (B.std(0) > 1e-9)
        A, B = A[:, keep], B[:, keep]
        mu = np.concatenate([A, B]).mean(0)
        sd = np.concatenate([A, B]).std(0) + 1e-9
        A, B = (A - mu) / sd, (B - mu) / sd
        if k:
            # denoise in the shared subspace, as connectome fingerprinting does
            _, _, Vt = np.linalg.svd(np.concatenate([A, B]), full_matrices=False)
            A, B = A @ Vt[:k].T, B @ Vt[:k].T
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        S = A @ B.T
        n = len(A)
        rank = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
        print(f'  {tag:<26} top-1 = {(rank==0).mean()*100:5.1f} %   '
              f'pct-rank = {(1-rank/(n-1)).mean()*100:5.1f} %   '
              f'(chance {100/n:.1f} %)  n={n}', flush=True)

    fp(A, B, 'FULL ATLAS')
    fp(A, B, 'FULL ATLAS + PCA-16', k=16)
    P = A.shape[1] // len(atoms.ATOM_NAMES)
    Ar = A.reshape(len(A), -1, len(atoms.ATOM_NAMES))
    Br = B.reshape(len(B), -1, len(atoms.ATOM_NAMES))
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        fp(Ar[:, :, ai], Br[:, :, ai], f'atom {nm}')
