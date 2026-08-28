"""
Re-measuring intrinsic dimension with a validated estimator.

The earlier verdicts, that a PMSM bench sits at 8.2 of 12 and a UR5e at 17.6 of
31, came from an estimator of my own that returns 0.29 and 0.00 on real cases
whose true dimension is 1. They are withdrawn. This repeats the measurement with
two estimators from the literature, agreeing with each other being the check
that neither is being trusted alone:

    TwoNN     Facco et al. 2017, scale free, uses only the two nearest
              neighbours, so nothing is tuned by hand
    MLE       Levina and Bickel 2005, local, reported as the median over points
              at several neighbourhood sizes so scale sensitivity is visible

Both are known to bias upward on noisy data, so a low reading is trustworthy and
a high one is an upper bound.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manifold_local as ml

P = os.path.expanduser('~/probe')
MAXN = 12000


def prep(X):
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(1)]
    if len(X) > MAXN:
        X = X[::max(1, len(X) // MAXN)][:MAXN]
    w = np.sqrt(np.maximum(ml.channel_snr(X) - 0.02, 0.0))
    live = w > 1e-6
    if live.sum() < 2:
        live = np.ones(X.shape[1], bool); w = np.ones(X.shape[1])
    return ml._ranks(X)[:, live] * w[live], int(live.sum()), X.shape[1]


def measure(name, X, note=''):
    try:
        Z, dlive, dall = prep(X)
    except Exception as e:
        print(f'  {name:<28} failed ({e})'); return
    if len(Z) < 500:
        print(f'  {name:<28} too short ({len(Z)})'); return
    twonn = ml.intrinsic_dim_twonn(Z)
    from sklearn.neighbors import NearestNeighbors
    mles = []
    for k in (16, 32, 64):
        if len(Z) < 4 * k:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        dist, _ = nn.kneighbors(Z[::max(1, len(Z) // 1500)])
        mles.append(np.median([ml.local_dim_mle(r[1:]) for r in dist]))
    mle = float(np.median(mles)) if mles else float('nan')
    ratio = twonn / max(dlive, 1)
    verdict = ('BODY' if ratio < 0.35 else
               'thin cloud' if ratio < 0.6 else 'CLOUD')
    print(f'  {name:<28} d={dall:3d} live={dlive:3d}  TwoNN={twonn:5.2f}  '
          f'MLE={mle:5.2f}  ratio={ratio:4.2f}  {verdict:11s} {note}')
    return ratio


print('re-measured with TwoNN and Levina-Bickel MLE\n')

# CNC, single experiment and pooled
fs_ = sorted(glob.glob(os.path.join(P, 'cnc', 'experiment_*.csv')))
if fs_:
    d0 = pd.read_csv(fs_[0])
    num = [c for c in d0.columns if c != 'Machining_Process'
           and pd.to_numeric(d0[c], errors='coerce').notna().mean() > .9]
    act = [c for c in num if 'Actual' in c]
    measure('CNC one run, all channels', d0[num].to_numpy(float),
            '(closed-loop servo)')
    measure('CNC one run, actual only', d0[act].to_numpy(float), '')
    X = pd.concat([pd.read_csv(f)[num] for f in fs_]).to_numpy(float)
    measure('CNC 18 runs pooled', X, '(different feeds/materials)')

# PMSM, previously reported as 8.17 of 12
cache = os.path.join(HERE, '_pmsm_sessions.npz')
if os.path.exists(cache):
    z = np.load(cache)
    for i in (0, 1, 2):
        measure(f'PMSM session {i}', z['X'][i].astype(float),
                '(was reported 8.17)' if i == 0 else '')

# robot, previously reported as 17.6 of 31
fl = os.path.expanduser('~/kaggle_kernel/out/fleet/ur5e_u80_e6_s90.npz')
if os.path.exists(fl):
    zf = np.load(fl, allow_pickle=False)
    keys = [k for k in zf.files if k.startswith('X_0_')][:2]
    if keys:
        measure('UR5e robot, one unit',
                np.concatenate([zf[k] for k in keys]).astype(float),
                '(was reported 17.6)')
