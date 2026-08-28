"""
Intrinsic dimension screening: which machines actually have a body?

This is the go/no-go for the whole IOO construction and it costs seconds. If a
system's telemetry fills most of its channel count there is no low-dimensional
shape to find, no useful chart to build, and no picture worth drawing. Two
systems already measured say so plainly: a UR5e under independent multisine
excitation sits at 17.6 of 31, and the PMSM bench at 8.0 of 12.

The prediction being tested is that machines under CLOSED-LOOP CONTROL running
REPEATED cycles are different in kind, because a controller enforcing a setpoint
is a constraint, and a constraint removes a degree of freedom. A CNC mill
logging both commanded and actual axis positions should be the extreme case: the
servo makes actual track command, so half the channels are near-copies of the
other half.

Reported as the ratio dim/d, because that is what decides whether the idea
applies. Below about 0.3 there is a real body. Above about 0.6 there is a cloud.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manifold_local as ml

PROBE = int(os.environ.get('PROBE', 500))


def numeric(df, drop=()):
    out = []
    for c in df.columns:
        if c in drop:
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().mean() > 0.9 and s.std(skipna=True) > 1e-12:
            out.append(c)
    return out


def screen(name, X, note='', ks=None):
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(1)]
    if len(X) < 400 or X.shape[1] < 3:
        print(f'  {name:<26} too small ({X.shape})')
        return None
    # k must exceed the channel count or the local PCA cannot resolve the
    # neighbourhood at all, and the dimension is capped at k by construction
    d0 = X.shape[1]
    if ks is None:
        ks = [max(2 * d0, 48), max(4 * d0, 96)]
    row = []
    for k in ks:
        if len(X) < 3 * k:
            continue
        g = ml.local_geometry(X, k=k, n_probe=min(PROBE, len(X) // 3), seed=0)
        row.append((k, float(np.median(g['fields']['dim'])), g['d']))
    if not row:
        return None
    d_live = row[0][2]
    dims = [r[1] for r in row]
    best = float(np.median(dims))
    ratio = best / max(d_live, 1)
    verdict = ('BODY' if ratio < 0.35 else
               'thin cloud' if ratio < 0.6 else 'CLOUD')
    print(f'  {name:<26} n={len(X):6d}  channels={X.shape[1]:3d}  live={d_live:3d}'
          f'  dim={best:5.2f}  dim/d={ratio:4.2f}  {verdict:11s} {note}')
    return dict(name=name, n=len(X), d=X.shape[1], live=d_live, dim=best,
                ratio=ratio, per_k=row)


def main():
    P = os.path.expanduser('~/probe')
    res = []

    # --- CNC mill: closed-loop servo, repeated toolpath ------------------
    fs_ = sorted(glob.glob(os.path.join(P, 'cnc', 'experiment_*.csv')))
    if fs_:
        d0 = pd.read_csv(fs_[0])
        cols = numeric(d0, drop=('Machining_Process',))
        cmd = [c for c in cols if 'Command' in c]
        act = [c for c in cols if 'Actual' in c]
        for tag, use in (('cnc all channels', cols),
                         ('cnc actual only', act),
                         ('cnc command only', cmd)):
            X = pd.concat([pd.read_csv(f)[use] for f in fs_[:6]]).to_numpy(float)
            res.append(screen(tag, X, note='(repeated toolpath)'))

    # --- battery: cycle-to-cycle degradation -----------------------------
    fb = sorted(glob.glob(os.path.join(P, 'batt', '*.csv')))
    if fb:
        df = pd.read_csv(fb[0])
        idc = [c for c in df.columns if 'battery' in c.lower()]
        cols = [c for c in numeric(df) if c not in ('cycle', 'cycle_life')]
        if idc:
            for bid, g in list(df.groupby(idc[0]))[:3]:
                if len(g) > 400:
                    res.append(screen(f'battery {bid}', g[cols].to_numpy(float),
                                      note='(charge/discharge cycles)'))
        else:
            res.append(screen('battery pooled', df[cols].to_numpy(float)))

    # --- reference points already measured -------------------------------
    cache = os.path.join(HERE, '_pmsm_sessions.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        res.append(screen('PMSM motor bench', z['X'][0].astype(float),
                          note='(reference: known cloud)'))

    print('\nranked by dim/d, lowest first (lowest is most usable):')
    for r in sorted([r for r in res if r], key=lambda r: r['ratio']):
        print(f'  {r["ratio"]:4.2f}  {r["name"]}')


if __name__ == '__main__':
    main()
