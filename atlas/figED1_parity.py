"""
Extended Data Figure 1 -- the even/odd decomposition under time reversal.

Eight atoms are even (invariant under x(t) -> x(T-t)) to machine precision;
the signed Levy area is odd, flips sign exactly, and is the only atom that
sees the arrow of time. Measured on 40 real motor sessions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import atoms
import figstyle as fs


def main():
    fs.setup()
    z = np.load(os.path.join(HERE, '_pmsm_sessions.npz'))
    X_all = z['X']

    fwd, rev = [], []
    for u in range(len(X_all)):
        X = X_all[u].astype(np.float64)
        fwd.append(atoms.atlas_unit(X, n_cells=1, min_cell=50)[0])
        rev.append(atoms.atlas_unit(X[::-1].copy(), n_cells=1, min_cell=50)[0])
    F, Rv = np.array(fwd), np.array(rev)

    fig = plt.figure(figsize=(9.2, 3.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.32, left=0.09, right=0.97,
                  top=0.80, bottom=0.20)

    ax = fig.add_subplot(gs[0, 0])
    names, same, flip = [], [], []
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        a, b = F[..., ai], Rv[..., ai]
        m = np.isfinite(a) & np.isfinite(b)
        names.append(nm)
        same.append(float(np.max(np.abs(b[m] - a[m]))))
        flip.append(float(np.max(np.abs(b[m] + a[m]))))
    same, flip = np.array(same), np.array(flip)
    x = np.arange(len(names))
    colors = [fs.C['red'] if flip[i] < same[i] * 1e-3 else fs.C['grey']
              for i in range(len(names))]
    ax.bar(x, same, color=colors, width=0.62)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('max |atom(reversed) - atom(forward)|')
    ax.set_ylim(1e-16, 1)
    ax.axhline(1e-12, color=fs.C['black'], ls=':', lw=1)
    ax.text(8.3, 1.4e-12, 'machine precision', fontsize=7, ha='right')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=fs.C['red'], label='odd (sees time)'),
                       Patch(color=fs.C['grey'], label='even (blind)')],
              fontsize=7.5, loc='lower left', frameon=False)
    ax.set_title('a   Parity under time reversal', loc='left', fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    li = atoms.ATOM_NAMES.index('levy')
    # atlas is (sessions, pairs, atoms); reduce over pairs, per session
    mag = np.nanmax(np.abs(F[..., li]), axis=1)
    ax.hist(mag, bins=24, color=fs.C['blue'], alpha=0.85)
    ax.axvline(np.median(mag), color=fs.C['red'], ls='--', lw=1.3)
    ax.text(np.median(mag), ax.get_ylim()[1] * 0.92,
            f'median {np.median(mag):.3f}', fontsize=7.5, color=fs.C['red'])
    ax.set_xlabel('max |levy| over pairs, per session')
    ax.set_ylabel('sessions')
    ax.set_title('b   The odd atom is not a curiosity:\nit carries real mass',
                 loc='left', fontsize=9.5)

    fig.suptitle('Exactly one of nine atoms can distinguish a lag from a lead',
                 y=0.97, fontsize=10.5)
    for ext in ('png', 'pdf'):
        out = os.path.join(HERE, f'figED1_parity.{ext}')
        fig.savefig(out, dpi=300 if ext == 'png' else None, format=ext)
    print('wrote figED1_parity.png + .pdf')


if __name__ == '__main__':
    main()
