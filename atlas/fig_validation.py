"""
Figure 2 -- validation: the geometry estimator recovers known shapes.

Tests on synthetic geometry with known answers: flat planes, curves, spheres,
Swiss rolls, slotted sheets, and the critical extensibility test (dead channels
must not change the dimension).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import figstyle as fs
import manifold_local as ml

rng = np.random.default_rng(0)
N = 5000
PROBE = 900


def measure(X, k=48, embed='z', snr=False):
    geo = ml.local_geometry(X, k=k, n_probe=PROBE, seed=0, embed=embed,
                            snr_weight=snr)
    f = geo['fields']
    return (np.median(f['dim']), np.median(f['curv']), np.median(f['tear']))


def main():
    fs.setup()
    fig = plt.figure(figsize=(12.8, 5.8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.06, right=0.98, top=0.84, bottom=0.10)

    cases = []

    # 1. flat 2-plane in R^5
    A = rng.normal(size=(N, 2))
    E = np.linalg.qr(rng.normal(size=(5, 5)))[0][:, :2]
    X = A @ E.T + 0.001 * rng.normal(size=(N, 5))
    d, c, t = measure(X)
    cases.append(('flat 2-plane\nin $\\mathbb{R}^5$', d, c, t, '2', '0'))

    # 2. 2-sphere, R=1 and R=4
    for R in (1.0, 4.0):
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        X = R * v
        d, c, t = measure(X, embed='none')
        cases.append((f'2-sphere\n$R={R:g}$', d, c, t, '2', f'1/{R:g}'))

    # 3. Swiss roll
    u = rng.uniform(1.5 * np.pi, 4.5 * np.pi, N)
    h = rng.uniform(0, 21, N)
    S = np.stack([u * np.cos(u), h, u * np.sin(u)], 1)
    d, c, t = measure(S)
    cases.append(('Swiss roll\n(curved sheet)', d, c, t, '2', 'curved'))

    # 4. slot cut in a sheet
    P2 = rng.uniform(-1, 1, size=(N * 2, 2))
    P2 = P2[~((np.abs(P2[:, 0]) < 0.25) & (P2[:, 1] > -0.3))][:N]
    slot = np.stack([P2[:, 0], P2[:, 1], 0.01 * rng.normal(size=len(P2))], 1)
    g_slot = ml.local_geometry(slot, k=48, n_probe=PROBE, seed=0, embed='z',
                              snr_weight=False)
    q = slot[g_slot['probe']]
    edge = (np.abs(np.abs(q[:, 0]) - 0.25) < 0.12) & (q[:, 1] > -0.2)
    interior = (np.abs(q[:, 0]) > 0.6)
    te = g_slot['fields']['tear']
    cases.append(('sheet with\na slot cut', measure(slot)[0], 0,
                 np.median(te[edge]), '2', 'tear at cut'))

    # Panel a-d: shapes with their measured dimension
    shape_data = [
        ('flat 2-plane', A @ E.T + 0.001 * rng.normal(size=(N, 5)),
         cases[0][1], 'z'),
        ('2-sphere R=1', 1.0 * v, cases[2][1], 'none'),
        ('Swiss roll', S, cases[3][1], 'z'),
        ('slotted sheet', slot, cases[4][3], 'z'),
    ]

    for i, (name, X, val, emb) in enumerate(shape_data):
        ax = fig.add_subplot(gs[0, i])
        Z = ml._ranks(X) if emb == 'z' else X - X.mean(0)
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        E2 = (Z - Z.mean(0)) @ Vt[:2].T
        ax.scatter(E2[::3, 0], E2[::3, 1], s=1, alpha=0.2, c=fs.C['blue'],
                   lw=0, rasterized=True)
        ax.set_title(f'{"abcd"[i]}   {name}\n     dim={val:.2f}',
                     loc='left', fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

    # Panel e: curvature scaling
    ax = fig.add_subplot(gs[1, 0])
    Rs = np.array([1, 2, 4, 8])
    curvs = []
    for R in Rs:
        v2 = rng.normal(size=(N, 3))
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
        d, c, t = measure(R * v2, embed='none')
        curvs.append(c)
    ax.bar(np.arange(len(Rs)), curvs, color=fs.C['green'], width=0.6)
    ax.plot(np.arange(len(Rs)), 1.0 / Rs, 'o-', color=fs.C['black'],
            label='$1/R$ (theory)', lw=1.5, ms=4)
    ax.set_xticks(np.arange(len(Rs)))
    ax.set_xticklabels([f'R={R}' for R in Rs], fontsize=8)
    ax.set_ylabel('curvature')
    ax.legend(fontsize=7)
    ax.set_title('e   curvature scales as 1/R', loc='left', fontsize=9)

    # Panel f: extensibility (dead channels)
    ax = fig.add_subplot(gs[1, 1])
    tt = np.linspace(0, 60, N)
    uu = 2.0 + 1.3 * np.sin(0.7 * tt) + 0.9 * np.sin(0.11 * tt)
    hh = 3.0 * np.sin(0.05 * tt)
    T3 = np.stack([uu * np.cos(uu), hh, uu * np.sin(uu)], 1)
    T3 = T3 + 0.01 * rng.normal(size=T3.shape)

    dims = []
    for extra in [0, 2, 6]:
        if extra == 0:
            Tx = T3
        else:
            Tx = np.concatenate([T3, 0.001 * rng.normal(size=(N, extra))], 1)
        d, c, t = measure(Tx, k=48, snr=True)
        dims.append(d)
    # live channel
    live = np.sin(0.31 * tt)[:, None] + 0.01 * rng.normal(size=(N, 1))
    Tl = np.concatenate([T3, live], 1)
    d, c, t = measure(Tl, k=48, snr=True)
    dims.append(d)

    labels = ['3 live\nonly', '+2 dead', '+6 dead', '+1 live']
    colors = [fs.C['blue'], fs.C['grey'], fs.C['grey'], fs.C['green']]
    ax.bar(np.arange(4), dims, color=colors, width=0.6)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('intrinsic dimension')
    ax.axhline(dims[0], color=fs.C['black'], ls=':', lw=1)
    ax.set_title('f   dead channels do not\n     change the body', loc='left',
                 fontsize=9)
    ax.set_ylim(0, 2.2)

    # Panel g: dimension by machining phase
    ax = fig.add_subplot(gs[1, 2])
    import pandas as pd
    P2_ = os.path.expanduser('~/probe/cnc')
    d0 = pd.read_csv(os.path.join(P2_, 'experiment_01.csv'))
    cols = [c for c in d0.columns if c != 'Machining_Process'
            and pd.to_numeric(d0[c], errors='coerce').notna().mean() > 0.9]
    allX, allP = [], []
    for i in range(1, 19):
        f = os.path.join(P2_, f'experiment_{i:02d}.csv')
        if not os.path.exists(f): continue
        d = pd.read_csv(f)
        X = d[cols].to_numpy(float)
        ok = np.isfinite(X).all(1)
        allX.append(X[ok])
        allP.append(d['Machining_Process'].values[ok])
    Xc = np.concatenate(allX)
    Pc = np.concatenate(allP)
    phases = pd.Series(Pc).value_counts()
    phase_dims = []
    phase_names = []
    for phase in phases.index[:6]:
        mask = (Pc == phase)
        if mask.sum() < 500: continue
        Xp = Xc[mask]
        Z = ml._ranks(Xp)
        w = np.sqrt(np.maximum(ml.channel_snr(Xp) - 0.02, 0.0))
        live = w > 1e-6
        if live.sum() < 3: continue
        Z = Z[:, live] * w[live]
        d = ml.intrinsic_dim_twonn(Z[::4])
        phase_dims.append(d)
        phase_names.append(phase.replace(' ', '\n', 1))
    ax.bar(np.arange(len(phase_dims)), phase_dims, color=fs.C['blue'],
           width=0.6)
    ax.axhline(35, color=fs.C['red'], ls='--', lw=1, label='35 live channels')
    ax.set_xticks(np.arange(len(phase_names)))
    ax.set_xticklabels(phase_names, fontsize=6.5, rotation=0)
    ax.set_ylabel('intrinsic dimension')
    ax.set_title('g   dimension by machining\n     phase (CNC)', loc='left',
                 fontsize=9)
    ax.legend(fontsize=6.5, loc='upper right')
    ax.set_ylim(0, 40)

    # Panel h: body as prior (denoising)
    ax = fig.add_subplot(gs[1, 3])
    sigmas = [0.02, 0.05, 0.10, 0.20]
    body = [0.387, 0.399, 0.438, 0.565]
    pca = [0.598, 0.613, 0.662, 0.824]
    median = [0.511, 0.533, 0.601, 0.776]
    x = np.arange(4)
    w = 0.25
    ax.bar(x - w, median, w, color=fs.C['grey'], label='median filter')
    ax.bar(x, pca, w, color=fs.C['blue'], label='linear subspace')
    ax.bar(x + w, body, w, color=fs.C['green'], label='body')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in sigmas], fontsize=8)
    ax.set_xlabel('noise $\\sigma$')
    ax.set_ylabel('restoration error (lower better)')
    ax.legend(fontsize=6.5, loc='upper left')
    ax.set_title('h   body as prior\n     beats linear by ~33%', loc='left',
                 fontsize=9)

    fig.suptitle('Validation: the estimator recovers known geometry, '
                 'is invariant to dead channels, and the body beats linear '
                 'denoising', y=0.93, fontsize=10.5)

    for ext in ('png', 'pdf'):
        out = os.path.join(HERE, f'fig2_validation.{ext}')
        fig.savefig(out, dpi=300 if ext == 'png' else None,
                    format=ext)
    print('wrote fig2_validation.png + .pdf')


if __name__ == '__main__':
    main()
