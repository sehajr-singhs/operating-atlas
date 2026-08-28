"""
How should the atlas be pooled?

The cell sweep showed a contradiction worth resolving: conditioning on operating
point improves every individual atom (levy 32.5 -> 42.5 % top-1) while making
the concatenated atlas WORSE (30 -> 17.5 %). Concatenation is the culprit, not
conditioning. Raw concatenation lets a block with many dimensions and a poor
signal-to-noise ratio dominate the cosine, and adding cells multiplies the
number of such blocks.

The atlas is naturally a set of blocks -- one per (cell, atom) -- each of which
is a comparable quantity across units. Pooling SIMILARITIES per block and then
combining is the right estimator: every block votes on the same footing
regardless of how many pairs it happens to contain.

Strategies compared:
  concat      the naive baseline, all numbers in one vector
  blockz      z-score each block, then concatenate (equalises block scale)
  simmean     mean of per-block cosine similarity matrices
  simz        mean of per-block similarity matrices, each z-scored across
              candidates first, so a block with a flat similarity profile
              cannot swamp a discriminative one
"""

import os
import time
import numpy as np
import pandas as pd
import atoms

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
PMSM_ALL = ['u_q', 'u_d', 'i_q', 'i_d', 'motor_speed', 'torque', 'ambient',
            'coolant', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
TAU_COLS = [PMSM_ALL.index(c) for c in ('i_q', 'i_d', 'torque')]
N = 16000
NA = len(atoms.ATOM_NAMES)


def load_sessions():
    df = pd.read_csv(os.path.join(ROOT, 'pmsm', 'measures_v2.csv'))
    return [(int(p), d[PMSM_ALL].to_numpy(np.float64)[:N])
            for p, d in df.groupby('profile_id', sort=True) if len(d) >= N]


def ident_rate(S):
    n = len(S)
    rank = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    return (rank == 0).mean() * 100, (1 - rank / (n - 1)).mean() * 100


def cos_sim(A, B):
    keep = (A.std(0) > 1e-9) & (B.std(0) > 1e-9)
    if keep.sum() < 2:
        return None
    A, B = A[:, keep], B[:, keep]
    pool = np.concatenate([A, B])
    mu, sd = pool.mean(0), pool.std(0) + 1e-9
    A, B = (A - mu) / sd, (B - mu) / sd
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def zrows(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)


if __name__ == '__main__':
    t0 = time.time()
    sess = load_sessions()
    print(f'{len(sess)} PMSM sessions x {N} samples', flush=True)

    for n_cells in [1, 3, 5]:
        A, B = [], []
        for pid, X in sess:
            Xa, Xb = X[:N // 2], X[N // 2:]
            ca = atoms.rank_load(Xa, TAU_COLS) if n_cells > 1 else None
            cb = atoms.rank_load(Xb, TAU_COLS) if n_cells > 1 else None
            A.append(atoms.atlas_unit(Xa, n_cells=n_cells, cell_coord=ca, min_cell=50))
            B.append(atoms.atlas_unit(Xb, n_cells=n_cells, cell_coord=cb, min_cell=50))
        A = np.nan_to_num(np.array(A))       # (u, cell, pair, atom)
        B = np.nan_to_num(np.array(B))
        u = len(A)
        print(f'\n=== n_cells={n_cells}  atlas {A.shape}  [{time.time()-t0:.0f}s] ===',
              flush=True)

        res = {}
        res['concat'] = cos_sim(A.reshape(u, -1), B.reshape(u, -1))

        blocks_a, blocks_b, sims = [], [], []
        for c in range(A.shape[1]):
            for ai in range(NA):
                a, b = A[:, c, :, ai], B[:, c, :, ai]
                sd = np.concatenate([a, b]).std(0) + 1e-9
                blocks_a.append((a - a.mean(0)) / sd)
                blocks_b.append((b - b.mean(0)) / sd)
                s = cos_sim(a, b)
                if s is not None:
                    sims.append(s)
        res['blockz'] = cos_sim(np.concatenate(blocks_a, 1),
                                np.concatenate(blocks_b, 1))
        res['simmean'] = np.mean(sims, axis=0)
        res['simz'] = np.mean([zrows(s) for s in sims], axis=0)

        for k, S in res.items():
            if S is None:
                continue
            t1, pr = ident_rate(S)
            print(f'  {k:<10} top-1 = {t1:5.1f} %   pct-rank = {pr:5.1f} %   '
                  f'(chance {100/u:.1f} %)', flush=True)

        # per-atom, pooled over cells with the winning strategy
        print('  per-atom (simz over cells):')
        for ai, nm in enumerate(atoms.ATOM_NAMES):
            ss = [zrows(cos_sim(A[:, c, :, ai], B[:, c, :, ai]))
                  for c in range(A.shape[1])
                  if cos_sim(A[:, c, :, ai], B[:, c, :, ai]) is not None]
            if not ss:
                continue
            t1, pr = ident_rate(np.mean(ss, axis=0))
            print(f'    {nm:>6} top-1 = {t1:5.1f} %   pct-rank = {pr:5.1f} %')
