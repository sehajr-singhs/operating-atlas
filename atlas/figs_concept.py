"""
Figure 1 -- what a machine's operating shapes look like, and what the atlas
reduces them to.

Two selection rules, both applied rather than described:
  * the session is the one with the richest shape content in the benchmark, not
    the longest. The longest is a staircase step-test whose largest hysteresis
    is 0.04, and drawing it would have illustrated the opposite of the point.
  * the exemplar pairs are the arg-max of one atom each, so the panels show
    what the method found rather than what the author wanted it to find.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import figstyle as fs            # noqa: E402
import atoms                     # noqa: E402

CACHE = os.path.join(HERE, '_pmsm_sessions.npz')
NICE = {'u_q': '$u_q$', 'u_d': '$u_d$', 'i_q': '$i_q$', 'i_d': '$i_d$',
        'motor_speed': 'speed', 'torque': 'torque', 'ambient': 'ambient',
        'coolant': 'coolant', 'pm': 'magnet $T$', 'stator_yoke': 'yoke $T$',
        'stator_tooth': 'tooth $T$', 'stator_winding': 'winding $T$'}
SHOW_ATOMS = [('eta', 'single-valued curve'),
              ('levy', 'oriented hysteresis loop'),
              ('nlgap', 'non-monotone branch'),
              ('jump', 'discontinuous coupling')]


def main():
    fs.setup()
    z = np.load(CACHE)
    X_all, pids, CH = z['X'], z['pids'], [str(c) for c in z['channels']]
    ex = np.load(os.path.join(HERE, '_exemplar.npz'))
    idx, pid = int(ex['idx']), int(ex['pid'])
    X = X_all[idx].astype(np.float64)
    d = X.shape[1]
    iu, ju = atoms.pair_index(d)
    A = atoms.atlas_unit(X, n_cells=1, min_cell=50)[0]

    show, used = [], set()
    for nm, _ in SHOW_ATOMS:
        ai = atoms.ATOM_NAMES.index(nm)
        v = np.where(np.isfinite(A[:, ai]), np.abs(A[:, ai]), -np.inf)
        for k in np.argsort(-v):
            if k not in used:
                used.add(k); show.append((k, nm, ai)); break

    fig = plt.figure(figsize=(11.6, 9.4))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[0.70, 1.05, 0.42, 0.95],
                  hspace=0.70, wspace=0.34,
                  left=0.075, right=0.90, top=0.925, bottom=0.065)

    # --- a,b the conventional view: two time scales ----------------------
    t = np.arange(len(X)) / 2.0 / 60.0
    for col, (title, chans, cols) in enumerate([
            ('mechanical: seconds', ['torque', 'motor_speed'],
             [fs.C['blue'], fs.C['sky']]),
            ('thermal: tens of minutes', ['stator_winding', 'coolant'],
             [fs.C['red'], fs.C['green']])]):
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2])
        for c, cc in zip(chans, cols):
            v = X[:, CH.index(c)]
            ax.plot(t, (v - v.mean()) / (v.std() + 1e-9), color=cc, lw=0.8,
                    label=NICE[c], rasterized=True)
        ax.set_xlabel('time (min)'); ax.set_ylabel('standardised')
        ax.set_title(title, loc='left')
        ax.legend(ncol=2, loc='upper right')
        fs.panel_label(ax, 'ab'[col], dx=-0.11)

    # --- c the shapes ----------------------------------------------------
    for p, (k, nm, ai) in enumerate(show):
        ax = fig.add_subplot(gs[1, p])
        i, j = iu[k], ju[k]
        s = max(1, len(X) // 3000)
        xs, ys = X[::s, i], X[::s, j]
        ax.plot(xs, ys, color=fs.C['grey'], lw=0.35, alpha=0.45,
                rasterized=True, zorder=1)
        sc = ax.scatter(xs, ys, c=np.arange(len(xs)), cmap='viridis', s=3.0,
                        lw=0, alpha=0.75, rasterized=True, zorder=2)
        ax.set_xlabel(NICE[CH[i]], labelpad=1.5)
        ax.set_ylabel(NICE[CH[j]], labelpad=1.5)
        ax.set_title(f'{nm} = {A[k, ai]:+.2f}', loc='left', fontsize=9,
                     color=fs.C['black'], pad=3)
        ax.text(0.0, 1.20, SHOW_ATOMS[p][1], transform=ax.transAxes,
                fontsize=7.8, color=fs.C['grey'], ha='left')
        ax.tick_params(labelsize=6.5)
        if p == 0:
            fs.panel_label(ax, 'c', dx=-0.30, dy=1.34)
    cax = fig.add_axes([0.915, 0.545, 0.009, 0.16])
    cb = fig.colorbar(sc, cax=cax); cb.set_label('time', fontsize=7.5)
    cb.set_ticks([])

    # --- c' the nine atoms for each exemplar pair ------------------------
    for p, (k, nm, ai) in enumerate(show):
        ax = fig.add_subplot(gs[2, p])
        vals = A[k].copy()
        norm = np.array([1, 1, 1, 1, 1, 1, 4.0, 1, 4.0])   # tau, beta are wider
        v = np.clip(vals / norm, -1, 1)
        cols = [fs.C['red'] if atoms.ATOM_NAMES[q] == nm else fs.C['grey']
                for q in range(len(v))]
        ax.bar(range(len(v)), v, color=cols, width=0.72)
        ax.axhline(0, color=fs.C['black'], lw=0.6)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(atoms.ATOM_NAMES, rotation=90, fontsize=5.8)
        ax.set_ylim(-1.08, 1.08); ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=6)
        if p == 0:
            ax.set_ylabel('atom', fontsize=7.5)

    # --- d the atlas -----------------------------------------------------
    ax = fig.add_subplot(gs[3, :])
    Z = np.zeros_like(A)
    for ai in range(A.shape[1]):
        v = A[:, ai]; f = np.isfinite(v)
        if f.sum() < 2:
            continue
        lo, hi = np.nanpercentile(v[f], [2, 98])
        Z[:, ai] = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    im = ax.imshow(Z.T, aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_yticks(range(len(atoms.ATOM_NAMES)))
    ax.set_yticklabels(atoms.ATOM_NAMES, fontsize=7.5)
    ax.set_xlabel(f"channel pair  ({len(iu)} pairs from {d} channels)", labelpad=12)
    ax.grid(False)
    ax.text(0.0, 1.13, 'the operating atlas of one unit, every pair and every '
                       'atom. This array is the "index of operations"',
            transform=ax.transAxes, fontsize=9, ha='left')
    fs.panel_label(ax, 'd', dx=-0.055, dy=1.20)
    # exemplar markers go BELOW the heatmap, where nothing else competes for
    # the space; above they collided with the panel title
    for p, (k, nm, ai) in enumerate(show):
        ax.axvline(k, color='white', lw=1.0, alpha=0.9)
        ax.annotate(f'c{p+1}', xy=(k, len(atoms.ATOM_NAMES) - 0.4),
                    xytext=(0, -13), textcoords='offset points',
                    ha='center', va='top', fontsize=7.5,
                    annotation_clip=False, color=fs.C['red'], fontweight='bold')
    cax2 = fig.add_axes([0.915, 0.075, 0.009, 0.17])
    cb2 = fig.colorbar(im, cax=cax2)
    cb2.set_label('atom value\n(per-atom scaled)', fontsize=7)
    cb2.set_ticks([])

    fig.suptitle(f'PMSM test bench, session {pid}: the same telemetry as time '
                 f'series, as shapes, and as an atlas', y=0.972, fontsize=11)
    out = os.path.join(HERE, 'fig1_concept.png')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.splitext(out)[0] + '.' + ext,
                    dpi=300 if ext == 'png' else None, format=ext)
        print('wrote', os.path.splitext(out)[0] + '.' + ext)
    print('wrote', out, f'(session {pid})')
    for p, (k, nm, ai) in enumerate(show):
        print(f'  c{p+1}: {CH[iu[k]]:>14} vs {CH[ju[k]]:<14} {nm} = {A[k, ai]:+.3f}')


if __name__ == '__main__':
    main()
