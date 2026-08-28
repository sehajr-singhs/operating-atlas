"""
Does the local geometry make the physics legible?

This is the claim the whole construction is actually for, and it is not a
retrieval benchmark. If the body's local geometry is a real description of the
machine, then the places where the geometry changes should be the places where
the physics changes, and a reader should be able to see it.

The robot testbed records a ground-truth regime at every sample, never shown to
the estimator:

    0  slow, nominal        2  slow, thermally derated
    1  fast, nominal        3  fast, derated
    4  drive tripped, holding brake

Those labels are a statement about the physics: derating means the achievable
torque now depends on winding temperature, so a constraint has tightened and a
degree of freedom has gone; a trip means the drive has switched off and the
mechanical subsystem has been cut loose from the electrical one, which is a
different body altogether, reached across a discontinuity.

So there are three concrete predictions, and they can each be wrong:

  P1  local dimension DROPS when a regime constrains the machine more, because
      a live constraint removes a degree of freedom
  P2  tear rises at regime BOUNDARIES, because a switch is a discontinuity in
      the body rather than a bend in it
  P3  the geometry alone separates the regimes, without ever being told them
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import figstyle as fs         # noqa: E402
import manifold_local as ml   # noqa: E402

FLEET = os.environ.get(
    'FLEET', os.path.expanduser('~/kaggle_kernel/out/fleet/ur5e_u80_e6_s90.npz'))
REG = ['slow nominal', 'fast nominal', 'slow derated', 'fast derated',
       'drive tripped']


def main():
    fs.setup()
    z = np.load(FLEET, allow_pickle=False)
    # pick the unit whose episodes visit the most regimes, so the picture has
    # something to show rather than one flat operating condition
    best, best_u = -1, None
    for k in z.files:
        if not k.startswith('L_'):
            continue
        nreg = len(np.unique(z[k]))
        if nreg > best:
            best, best_u = nreg, k[2:]
    u, e = best_u.split('_')
    X = np.concatenate([z[f'X_{u}_{ep}'] for ep in range(6)
                        if f'X_{u}_{ep}' in z.files]).astype(np.float64)
    L = np.concatenate([z[f'L_{u}_{ep}'] for ep in range(6)
                        if f'L_{u}_{ep}' in z.files])
    print(f'unit {u}: {X.shape[0]} samples, {X.shape[1]} channels, '
          f'regimes present {np.unique(L)}')

    step = max(1, len(X) // 30000)
    X, L = X[::step], L[::step]
    geo = ml.local_geometry(X, k=64, n_probe=6000, seed=0)
    p = geo['probe']
    lab = L[p]
    F = geo['fields']

    print('\nP1  local dimension by regime (never shown to the estimator)')
    for r in np.unique(lab):
        m = lab == r
        if m.sum() < 30:
            continue
        print(f'   {REG[r]:<16} n={m.sum():5d}   dim {np.median(F["dim"][m]):5.2f}'
              f'   curv {np.median(F["curv"][m]):7.3f}'
              f'   tear {np.median(F["tear"][m]):5.2f}'
              f'   v_norm {np.median(F["v_norm"][m]):.4f}')

    print('\nP2  tear at regime boundaries versus inside a regime')
    switch = np.zeros(len(L), bool)
    switch[1:] = L[1:] != L[:-1]
    win = 25
    near = np.convolve(switch.astype(float), np.ones(2 * win + 1), 'same') > 0
    nb, inb = near[p], ~near[p]
    if nb.sum() > 20 and inb.sum() > 20:
        print(f'   within {win} samples of a switch: tear '
              f'{np.median(F["tear"][nb]):.3f}  (n={nb.sum()})')
        print(f'   away from any switch:          tear '
              f'{np.median(F["tear"][inb]):.3f}  (n={inb.sum()})')
        print(f'   off-manifold speed near a switch '
              f'{np.median(F["v_norm"][nb]):.4f} vs away '
              f'{np.median(F["v_norm"][inb]):.4f}')

    print('\nP3  can the geometry alone tell the regimes apart?')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    G = ml.field_matrix(geo)
    keep = np.isfinite(G).all(1)
    G, y = G[keep], lab[keep]
    ok = np.array([c for c in np.unique(y) if (y == c).sum() >= 40])
    m = np.isin(y, ok)
    G, y = G[m], y[m]
    if len(ok) >= 2:
        clf = RandomForestClassifier(n_estimators=250, random_state=0,
                                     min_samples_leaf=3, n_jobs=-1)
        s = cross_val_score(clf, G, y, cv=StratifiedKFold(5, shuffle=True,
                                                          random_state=0))
        maj = max((y == c).mean() for c in ok)
        print(f'   {len(ok)} regimes, {s.mean()*100:.1f} +- {s.std()*100:.1f} % '
              f'from 7 geometric numbers alone  (majority {maj*100:.1f} %)')
        clf.fit(G, y)
        for nm, imp in sorted(zip(ml.FIELD_NAMES, clf.feature_importances_),
                              key=lambda t: -t[1]):
            print(f'      {nm:<8} {imp:.3f}')

    # ---- the picture ----------------------------------------------------
    Zp = geo['Z'][p]
    _, _, Vt = np.linalg.svd(Zp - Zp.mean(0), full_matrices=False)
    E = (Zp - Zp.mean(0)) @ Vt[:3].T

    fig = plt.figure(figsize=(13.0, 6.6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.28, wspace=0.22,
                  left=0.03, right=0.98, top=0.86, bottom=0.06)
    panels = [('dim', 'local dimension', 'viridis'),
              ('curv', 'curvature', 'magma'),
              ('tear', 'tear', 'cividis')]
    for c, (key, ttl, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c], projection='3d')
        v = F[key]
        lo, hi = np.nanpercentile(v, [3, 97])
        sc = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=np.clip(v, lo, hi),
                        cmap=cmap, s=1.8, lw=0, alpha=.7)
        ax.set_title(f'{"abc"[c]}   {ttl}', loc='left', fontsize=9.5)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_ticklabels([])
        ax.grid(False)
        fig.colorbar(sc, ax=ax, shrink=.5, pad=.02).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[0, 3], projection='3d')
    cols = [fs.C['sky'], fs.C['blue'], fs.C['orange'], fs.C['red'],
            fs.C['purple']]
    for r in np.unique(lab):
        m2 = lab == r
        ax.scatter(E[m2, 0], E[m2, 1], E[m2, 2], s=1.8, lw=0, alpha=.7,
                   color=cols[r], label=REG[r])
    ax.set_title('d   ground-truth regime\n     (never shown)', loc='left',
                 fontsize=9.5)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticklabels([])
    ax.grid(False)
    ax.legend(fontsize=6, loc='upper left', markerscale=3)

    for c, key in enumerate(['dim', 'curv', 'tear', 'v_norm']):
        ax = fig.add_subplot(gs[1, c])
        data, names, colr = [], [], []
        for r in np.unique(lab):
            m2 = lab == r
            if m2.sum() < 30:
                continue
            data.append(F[key][m2]); names.append(REG[r].split()[0][:5])
            colr.append(cols[r])
        bp = ax.boxplot(data, labels=names, showfliers=False, patch_artist=True,
                        widths=.6)
        for patch, cc in zip(bp['boxes'], colr):
            patch.set_facecolor(cc); patch.set_alpha(.65)
            patch.set_edgecolor(fs.C['black'])
        for med in bp['medians']:
            med.set_color(fs.C['black'])
        ax.set_title(f'{"efgh"[c]}   {key} by regime', loc='left', fontsize=9.5)
        ax.tick_params(labelsize=7)

    fig.suptitle('The operating body of one robot joint set, and whether its '
                 'local geometry recovers physics it was never told',
                 y=0.955, fontsize=11)
    out = os.path.join(HERE, 'fig6_legible.png')
    fig.savefig(out, dpi=190)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
