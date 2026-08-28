"""
Cache the PMSM sessions once, then settle two questions in one controlled pass.

Q1  HOW SHOULD CELLS BE DEFINED?
    Two earlier runs disagreed about whether conditioning on operating point
    helps, and they differed in more than one place, so neither settles it.
    Here the split, the data and the code path are identical and only the cell
    definition changes:
        none      one atlas over the whole record
        global    quantiles of raw torque, edges fitted once on the pooled
                  fleet -- an ABSOLUTE operating point, which presumes every
                  machine is calibrated the same way
        rank      quantiles of the mean per-channel rank of the torque group --
                  a RELATIVE operating point, exactly invariant to per-channel
                  recalibration
    The expectation from both earlier runs is that `global` wins on
    identification and `rank` is the only one that survives re-instrumentation.
    If so the trade-off is real and has to be reported rather than resolved.

Q2  WHICH SESSION IS WORTH DRAWING?
    The first attempt at the concept figure used the longest session, which
    turned out to be a staircase step-test: piecewise-constant torque, so every
    pair plots as vertical stripes and the largest hysteresis in the whole
    session was 0.04. Sessions are ranked by shape richness instead.
"""

import os
import time
import numpy as np
import pandas as pd
import atoms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'data')
CACHE = os.path.join(HERE, '_pmsm_sessions.npz')
CH = ['u_q', 'u_d', 'i_q', 'i_d', 'motor_speed', 'torque', 'ambient',
      'coolant', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
TAU_GROUP = [CH.index(c) for c in ('i_q', 'i_d', 'torque')]
TORQUE = CH.index('torque')
N = 16000


def build_cache():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z['X'], z['pids']
    df = pd.read_csv(os.path.join(ROOT, 'pmsm', 'measures_v2.csv'))
    Xs, pids = [], []
    for p, d in df.groupby('profile_id', sort=True):
        if len(d) < N:
            continue
        Xs.append(d[CH].to_numpy(np.float32)[:N]); pids.append(int(p))
    X = np.stack(Xs)
    np.savez_compressed(CACHE, X=X, pids=np.array(pids), channels=np.array(CH))
    return X, np.array(pids)


def ident(S):
    n = len(S)
    r = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    return (r == 0).mean() * 100, (1 - r / (n - 1)).mean() * 100


def cos_sim(A, B):
    keep = (A.std(0) > 1e-9) & (B.std(0) > 1e-9)
    if keep.sum() < 2:
        return None
    A, B = A[:, keep], B[:, keep]
    pool = np.concatenate([A, B]); mu, sd = pool.mean(0), pool.std(0) + 1e-9
    A, B = (A - mu) / sd, (B - mu) / sd
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def atlases(X, mode, n_cells, warp_seed=None):
    """X : (u, N, d) -> (A, B) atlases of the first and second half."""
    A, B = [], []
    edges = None
    if mode == 'global' and n_cells > 1:
        edges = atoms.fit_cell_edges(X[:, :, TORQUE].ravel(), n_cells)
    for u in range(len(X)):
        Xa = X[u, :N // 2].astype(np.float64)
        Xb = X[u, N // 2:].astype(np.float64)
        if warp_seed is not None:
            Xb = atoms.warp_channels(Xb, np.random.default_rng(warp_seed + u))
        if n_cells == 1:
            ca = cb = None
        elif mode == 'global':
            ca, cb = Xa[:, TORQUE], Xb[:, TORQUE]
        else:
            ca = atoms.rank_load(Xa, TAU_GROUP); cb = atoms.rank_load(Xb, TAU_GROUP)
        A.append(atoms.atlas_unit(Xa, n_cells=n_cells, cell_coord=ca,
                                  cell_edges=edges, min_cell=50))
        B.append(atoms.atlas_unit(Xb, n_cells=n_cells, cell_coord=cb,
                                  cell_edges=edges, min_cell=50))
    return np.nan_to_num(np.array(A)), np.nan_to_num(np.array(B))


if __name__ == '__main__':
    t0 = time.time()
    X, pids = build_cache()
    u = len(X)
    print(f'cached {X.shape} sessions [{time.time()-t0:.0f}s]', flush=True)

    LEVY = atoms.ATOM_NAMES.index('levy')
    print(f'\n{"mode":<8}{"cells":>6}{"atlas t1":>10}{"atlas pct":>11}'
          f'{"levy t1":>9}{"warp t1":>9}{"warp shift":>12}')
    for mode, nc in [('none', 1), ('global', 3), ('rank', 3),
                     ('global', 5), ('rank', 5)]:
        A, B = atlases(X, mode, nc)
        S = cos_sim(A.reshape(u, -1), B.reshape(u, -1))
        t1, pct = ident(S)
        sl = cos_sim(A[:, :, :, LEVY].reshape(u, -1), B[:, :, :, LEVY].reshape(u, -1))
        lt1, _ = ident(sl)
        # recalibrate the held-out half and repeat
        _, Bw = atlases(X, mode, nc, warp_seed=9000)
        inv = [i for i, nm in enumerate(atoms.ATOM_NAMES) if nm in atoms.WARP_INVARIANT]
        Sw = cos_sim(A[..., inv].reshape(u, -1), Bw[..., inv].reshape(u, -1))
        wt1, _ = ident(Sw)
        shift = float(np.nanmax(np.abs(B[..., inv] - Bw[..., inv])))
        print(f'{mode:<8}{nc:>6}{t1:>9.1f}%{pct:>10.1f}%{lt1:>8.1f}%'
              f'{wt1:>8.1f}%{shift:>12.2e}', flush=True)

    # ---- exemplar session for the concept figure ------------------------
    print('\nshape richness by session (for the concept figure):')
    rich = []
    for i in range(u):
        Ai = atoms.atlas_unit(X[i].astype(np.float64), n_cells=1, min_cell=50)[0]
        lv = np.nanpercentile(np.abs(Ai[:, LEVY]), 95)
        ng = np.nanpercentile(Ai[:, atoms.ATOM_NAMES.index('nlgap')], 95)
        rich.append((float(lv), float(ng), int(pids[i]), i))
    rich.sort(reverse=True)
    for lv, ng, p, i in rich[:6]:
        print(f'  session {p:>3}  p95|levy| = {lv:.3f}   p95 nlgap = {ng:.3f}')
    np.savez(os.path.join(HERE, '_exemplar.npz'),
             pid=rich[0][2], idx=rich[0][3])
    print(f'chosen session {rich[0][2]}  [{time.time()-t0:.0f}s]')
