"""
The operating body of a CNC mill, drawn.

47 logged channels, about 8 intrinsic dimensions, because the servos force
actual axis state to track commanded state and the part program forces the
path. That is why this one can be drawn at all and the earlier systems could
not: there is a shape here rather than a cloud filling its box.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import figstyle as fs         # noqa: E402
import manifold_local as ml   # noqa: E402

P = os.path.expanduser('~/probe/cnc')


def main():
    fs.setup()
    meta = pd.read_csv(os.path.join(P, 'train.csv'))
    d0 = pd.read_csv(os.path.join(P, 'experiment_01.csv'))
    cols = [c for c in d0.columns if c != 'Machining_Process'
            and pd.to_numeric(d0[c], errors='coerce').notna().mean() > 0.9]

    runs, feed, worn = [], [], []
    for i in range(1, 19):
        f = os.path.join(P, f'experiment_{i:02d}.csv')
        if not os.path.exists(f):
            continue
        X = pd.read_csv(f)[cols].to_numpy(float)
        X = X[np.isfinite(X).all(1)]
        runs.append(X)
        feed.append(meta.feedrate[i - 1])
        worn.append(meta.tool_condition[i - 1] == 'worn')

    X = runs[0]
    geo = ml.local_geometry(X, k=40, n_probe=1000, seed=0)
    p = geo['probe']
    Zp = geo['Z'][p]
    _, _, Vt = np.linalg.svd(Zp - Zp.mean(0), full_matrices=False)
    E = (Zp - Zp.mean(0)) @ Vt[:3].T

    fig = plt.figure(figsize=(12.8, 7.6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.26,
                  left=0.045, right=0.98, top=0.86, bottom=0.08)

    # a  the raw tool path, the thing everybody already draws
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    xyz = [c for c in ('X1_ActualPosition', 'Y1_ActualPosition',
                       'Z1_ActualPosition') if c in cols]
    T = pd.read_csv(os.path.join(P, 'experiment_01.csv'))[xyz].to_numpy(float)
    ax.plot(T[:, 0], T[:, 1], T[:, 2], lw=.5, color=fs.C['grey'])
    ax.set_title('a   the tool path in space\n     (3 of 47 channels)',
                 loc='left', fontsize=9)
    for A in (ax.xaxis, ax.yaxis, ax.zaxis):
        A.set_ticklabels([])
    ax.grid(False)

    # b,c,d  the OPERATING body, coloured by its own local geometry
    for c, (key, ttl, cmap) in enumerate([('dim', 'local dimension', 'viridis'),
                                          ('curv', 'curvature', 'magma'),
                                          ('v_tan', 'speed along the body',
                                           'cividis')]):
        ax = fig.add_subplot(gs[0, c + 1], projection='3d')
        v = geo['fields'][key]
        lo, hi = np.nanpercentile(v, [3, 97])
        sc = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=np.clip(v, lo, hi),
                        cmap=cmap, s=4, lw=0, alpha=.8)
        ax.set_title(f'{"bcd"[c]}   {ttl}', loc='left', fontsize=9)
        for A in (ax.xaxis, ax.yaxis, ax.zaxis):
            A.set_ticklabels([])
        ax.grid(False)
        fig.colorbar(sc, ax=ax, shrink=.5, pad=.02).ax.tick_params(labelsize=6)

    # e  intrinsic dimension: why this one is drawable and the others were not
    ax = fig.add_subplot(gs[1, 0])
    names = ['CNC mill', 'UR5e robot', 'PMSM bench']
    dims = [7.90, 9.12, 5.90]
    tot = [47, 31, 12]
    r = [d / t for d, t in zip(dims, tot)]
    cc = [fs.C['green'], fs.C['blue'], fs.C['red']]
    ax.barh(np.arange(3), r, color=cc, height=.6)
    for i, (d, t) in enumerate(zip(dims, tot)):
        ax.text(r[i] + .02, i, f'{d:.1f} of {t}', va='center', fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, .75)
    ax.set_xlabel('intrinsic dim / channels')
    ax.axvline(.35, color=fs.C['black'], ls=':', lw=1)
    ax.text(.355, 2.4, 'body   |   cloud', fontsize=7)
    ax.set_title('e   which machines have a body', loc='left', fontsize=9)

    # f  every run on the shared body, coloured by feedrate
    ax = fig.add_subplot(gs[1, 1:3])
    sub = [r_[::4] for r_ in runs]
    pooled = np.concatenate(sub)
    Zr = ml._ranks(pooled)
    _, _, Vt2 = np.linalg.svd(Zr - Zr.mean(0), full_matrices=False)
    off = 0
    cmap = plt.get_cmap('plasma')
    fmin, fmax = min(feed), max(feed)
    for r_, fv in zip(sub, feed):
        e = (Zr[off:off + len(r_)] - Zr.mean(0)) @ Vt2[:2].T
        off += len(r_)
        ax.scatter(e[:, 0], e[:, 1], s=1.2, lw=0, alpha=.35,
                   color=cmap((fv - fmin) / max(fmax - fmin, 1)))
    ax.set_xticks([]); ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=fmin, vmax=fmax))
    fig.colorbar(sm, ax=ax, shrink=.7, pad=.01,
                 label='feedrate').ax.tick_params(labelsize=6)
    ax.set_title('f   all 18 runs of the same program on one body.\n'
                 '     Feedrate moves where the run sits on it '
                 '(decoded at R2 = 0.91)', loc='left', fontsize=9)

    # g  what the geometry decodes, and what it does not
    ax = fig.add_subplot(gs[1, 3])
    lbl = ['feedrate', 'clamp\npressure', 'tool\nwear']
    g_ = [0.91, 0.00, 0.278 - 0.556]
    m_ = [0.69, 0.00, 0.50 - 0.556]
    x = np.arange(3); w = .36
    ax.bar(x - w / 2, g_, w, color=fs.C['green'], label='geometry')
    ax.bar(x + w / 2, m_, w, color=fs.C['grey'], label='marginals')
    ax.axhline(0, color=fs.C['black'], lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8)
    ax.set_ylabel('R$^2$, or accuracy above majority')
    ax.legend(fontsize=7)
    ax.set_title('g   one clear win, two failures', loc='left', fontsize=9)

    fig.suptitle('A CNC mill has an operating body: 47 logged channels, about 8 '
                 'intrinsic dimensions', y=.95, fontsize=11)
    for ext in ('png', 'pdf'):
        out = os.path.join(HERE, f'fig7_cnc.{ext}')
        fig.savefig(out, dpi=300 if ext == 'png' else None, format=ext)
    print('wrote fig7_cnc.png + .pdf')


if __name__ == '__main__':
    main()
