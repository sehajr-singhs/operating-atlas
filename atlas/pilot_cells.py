"""
Does conditioning on operating point help, and how much of the noise is
estimation versus drift?

Two things are tested at once on the real PMSM bench (40 sessions, >=16k
samples each).

1. CELLS. The claim under test is not that a machine has one shape but that its
   shape MOVES with operating point. Pooling a whole session into a single
   atlas superposes every regime and can average a clean curve into a splotch.
   n_cells is swept with cells defined by a load coordinate shared across all
   units (torque), edges fitted once on the pooled fleet.

2. SPLIT TYPE. Reproducibility measured on two disjoint windows conflates two
   different things. Comparing an INTERLEAVED split (even vs odd samples, same
   operating conditions) isolates pure estimation noise; comparing DISJOINT
   windows (first vs last, different operating conditions) adds genuine
   non-stationarity. The gap between them says whether more data would help or
   whether the machine has simply changed.
"""

import os
import time
import numpy as np
import pandas as pd
import atoms

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
PMSM_ALL = ['u_q', 'u_d', 'i_q', 'i_d', 'motor_speed', 'torque', 'ambient',
            'coolant', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
LOAD_CH = PMSM_ALL.index('torque')       # the load coordinate, shared by all units
N = 16000


def load_sessions(min_len=N):
    df = pd.read_csv(os.path.join(ROOT, 'pmsm', 'measures_v2.csv'))
    out = []
    for pid, d in df.groupby('profile_id', sort=True):
        if len(d) < min_len:
            continue
        out.append((int(pid), d[PMSM_ALL].to_numpy(np.float64)[:N]))
    return out


def splits(X, kind):
    if kind == 'interleaved':
        return X[0::2], X[1::2]
    return X[:len(X) // 2], X[len(X) // 2:]


def atom_corr(A, B):
    out = {}
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        x, y = A[..., ai].ravel(), B[..., ai].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
            out[nm] = np.nan
        else:
            out[nm] = float(np.corrcoef(x[m], y[m])[0, 1])
    return out


def fingerprint(A, B, tag, k=None):
    A = np.nan_to_num(np.asarray(A)); B = np.nan_to_num(np.asarray(B))
    keep = (A.std(0) > 1e-9) & (B.std(0) > 1e-9)
    A, B = A[:, keep], B[:, keep]
    if A.shape[1] < 2:
        print(f'  {tag:<34} degenerate'); return np.nan
    pool = np.concatenate([A, B])
    A = (A - pool.mean(0)) / (pool.std(0) + 1e-9)
    B = (B - pool.mean(0)) / (pool.std(0) + 1e-9)
    if k:
        _, _, Vt = np.linalg.svd(np.concatenate([A, B]), full_matrices=False)
        kk = min(k, Vt.shape[0])
        A, B = A @ Vt[:kk].T, B @ Vt[:kk].T
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = A @ B.T
    n = len(A)
    rank = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    t1 = (rank == 0).mean() * 100
    pr = (1 - rank / (n - 1)).mean() * 100
    print(f'  {tag:<34} top-1 = {t1:5.1f} %   pct-rank = {pr:5.1f} %   '
          f'(chance {100/n:.1f} % / 50 %)  dim={A.shape[1]}', flush=True)
    return t1


if __name__ == '__main__':
    t0 = time.time()
    sess = load_sessions()
    print(f'{len(sess)} PMSM sessions, {N} samples each, '
          f'{len(PMSM_ALL)} channels', flush=True)

    for kind in ['interleaved', 'disjoint']:
        for n_cells in [1, 3, 5]:
            # cell edges fitted ONCE on the pooled fleet, then applied to every
            # unit and every split, so 'cell 2' is the same operating condition
            # everywhere
            edges = None
            if n_cells > 1:
                pooled = np.concatenate([X[:, LOAD_CH] for _, X in sess])
                edges = atoms.fit_cell_edges(pooled, n_cells)

            AS, BS, corrs = [], [], []
            for pid, X in sess:
                Xa, Xb = splits(X, kind)
                ca = Xa[:, LOAD_CH] if n_cells > 1 else None
                cb = Xb[:, LOAD_CH] if n_cells > 1 else None
                A = atoms.atlas_unit(Xa, n_cells=n_cells, cell_coord=ca,
                                     cell_edges=edges, min_cell=50)
                B = atoms.atlas_unit(Xb, n_cells=n_cells, cell_coord=cb,
                                     cell_edges=edges, min_cell=50)
                AS.append(A.reshape(-1)); BS.append(B.reshape(-1))
                corrs.append(atom_corr(A, B))
            mc = {nm: np.nanmean([c[nm] for c in corrs]) for nm in atoms.ATOM_NAMES}
            print(f'\n[{kind}, n_cells={n_cells}]  reproducibility  '
                  + '  '.join(f'{nm}={mc[nm]:+.2f}' for nm in atoms.ATOM_NAMES)
                  + f'   [{time.time()-t0:.0f}s]', flush=True)
            fingerprint(AS, BS, 'FULL ATLAS')
            fingerprint(AS, BS, 'FULL ATLAS + PCA-16', k=16)
            AR = np.array(AS).reshape(len(AS), -1, len(atoms.ATOM_NAMES))
            BR = np.array(BS).reshape(len(BS), -1, len(atoms.ATOM_NAMES))
            best = [(fingerprint(AR[:, :, ai], BR[:, :, ai], f'atom {nm}'), nm)
                    for ai, nm in enumerate(atoms.ATOM_NAMES)]
    print(f'\ndone [{time.time()-t0:.0f}s]')
