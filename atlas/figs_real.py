"""
Figure 4 -- the real-hardware benchmark.

Six real industrial systems, nothing simulated except the recalibration warp,
and one experiment (the gas turbine) where even the drift is real.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import figstyle as fs        # noqa: E402
import atoms                 # noqa: E402

PRETTY = {'gasturbine': 'gas turbine', 'hydraulic': 'hydraulic rig',
          'transformer': 'transformer', 'wind': 'wind farm',
          'motor': 'motor bench', 'compressor': 'air compressor',
          'pumprig': 'pump rig'}
COMP = {'cooler': 'cooler', 'valve': 'valve', 'pump': 'pump leak',
        'acc': 'accumulator', 'stable': 'stable flag'}


def pct(v):
    return float('nan') if v is None or not np.isfinite(v) else v * 100


def main():
    fs.setup()
    R = json.load(open(os.path.join(HERE, 'results_real.json')))
    z = np.load(os.path.join(HERE, 'profiles_real.npz'), allow_pickle=False)
    P, Pw, L = z['P'], z['Pw'], np.array([str(x) for x in z['L']])

    fig = plt.figure(figsize=(12.6, 8.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.34,
                  left=0.07, right=0.985, top=0.875, bottom=0.10)

    # --- a class separation ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    r1 = R.get('R1', {})
    keys = [('atlas', 'atom profile', fs.C['red']),
            ('no_tau_fill', 'without tau, fill', fs.C['orange']),
            ('marginal', 'marginal profile', fs.C['grey']),
            ('rho', 'correlation only', fs.C['blue'])]
    vals = [(lab, pct(r1.get(k, [np.nan])[0]), c) for k, lab, c in keys
            if k in r1]
    y = np.arange(len(vals))
    ax.barh(y, [v for _, v, _ in vals], color=[c for _, _, c in vals], height=.6)
    for yy, (_, v, _) in zip(y, vals):
        if np.isfinite(v):
            ax.text(v + 1.2, yy, f'{v:.0f}%', va='center', fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([l for l, _, _ in vals], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('class accuracy (%)'); ax.set_xlim(0, 108)
    ax.set_title('a   which system is this?', loc='left', fontsize=9.5)

    # --- b embedding -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    Z = (P - P.mean(0)) / (P.std(0) + 1e-9)
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    E = Z @ Vt[:2].T
    pal = [fs.C['blue'], fs.C['orange'], fs.C['green'], fs.C['red'],
           fs.C['purple'], fs.C['sky'], fs.C['grey']]
    for k, c in enumerate(sorted(set(L))):
        m = L == c
        ax.scatter(E[m, 0], E[m, 1], s=20, color=pal[k % len(pal)], lw=.4,
                   edgecolor='white', label=PRETTY.get(c, c))
    ax.set_xlabel('profile PC1'); ax.set_ylabel('profile PC2')
    ax.legend(fontsize=6.2, loc='best')
    ax.set_title('b   real systems', loc='left', fontsize=9.5)

    # --- c recalibration ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    r4 = R.get('R4', {})
    before = [pct(r1.get('atlas', [np.nan])[0]), pct(r1.get('marginal', [np.nan])[0])]
    after = [pct(r4.get('atlas_warped', [np.nan])[0]),
             pct(r4.get('marginal_warped', [np.nan])[0])]
    x = np.arange(2); w = .34
    ax.bar(x - w / 2, before, w, color=fs.C['red'], label='as calibrated')
    ax.bar(x + w / 2, after, w, color=fs.C['sky'], label='recalibrated')
    for xx, b, a in zip(x, before, after):
        if np.isfinite(b): ax.text(xx - w / 2, b + 1.2, f'{b:.0f}', ha='center', fontsize=7.5)
        if np.isfinite(a): ax.text(xx + w / 2, a + 1.2, f'{a:.0f}', ha='center', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(['atom\nprofile', 'marginal\nprofile'],
                                         fontsize=8)
    ax.set_ylabel('class accuracy (%)'); ax.set_ylim(0, 112)
    ax.legend(fontsize=7, loc='lower left')
    sh = r4.get('max_invariant_shift', None)
    if sh is not None:
        ax.text(.02, .96, f'max shift on the eight\ninvariant atoms: {sh:.1e}',
                transform=ax.transAxes, fontsize=7, va='top', color=fs.C['grey'])
    ax.set_title('c   an independent warp per channel', loc='left', fontsize=9.5)

    # --- d real five-year drift -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    r2 = R.get('R2', {})
    if r2:
        v = [pct(r2.get('year_from_atlas', [np.nan])[0]),
             pct(r2.get('year_from_marginal', [np.nan])[0])]
        ax.bar([0, 1], v, color=[fs.C['red'], fs.C['grey']], width=.6)
        for xx, vv in zip([0, 1], v):
            if np.isfinite(vv):
                ax.text(xx, vv + 1.5, f'{vv:.0f}%', ha='center', fontsize=8)
        ax.axhline(20, color=fs.C['black'], lw=1, ls=':')
        ax.text(1.45, 21, 'chance', fontsize=7, ha='right')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['atom\nprofile', 'per-channel\nmarginals'],
                                                  fontsize=8)
        ax.set_ylabel('accuracy dating a segment (%)'); ax.set_ylim(0, 108)
        da, dm = r2.get('drift_atlas'), r2.get('drift_marginal')
        if da and dm:
            ax.text(.02, .96, f'2011 to 2015 centroid shift\natlas {da:.2f} vs '
                              f'marginals {dm:.2f}', transform=ax.transAxes,
                    fontsize=7, va='top', color=fs.C['grey'])
    ax.set_title('d   real drift: can you date a segment?\n      (being hard to '
                 'date is the good outcome)', loc='left', fontsize=9.5)

    # --- e fault diagnosis -------------------------------------------------
    ax = fig.add_subplot(gs[1, 1:])
    r3 = R.get('R3', {})
    comps = [c for c in ['cooler', 'valve', 'pump', 'acc'] if c in r3]
    if comps:
        x = np.arange(len(comps)); w = .2
        series = [('atlas', 'atlas', fs.C['red']),
                  ('atlas_warped', 'atlas, recalibrated', fs.C['orange']),
                  ('marginal', 'marginals', fs.C['grey']),
                  ('marginal_warped', 'marginals, recalibrated', fs.C['sky'])]
        for k, (key, lab, col) in enumerate(series):
            v = [pct(r3[c].get(key, [np.nan])[0]) for c in comps]
            ax.bar(x + (k - 1.5) * w, v, w, color=col, label=lab)
        # The majority-class rate is the honest floor here, and without it two
        # of these four components look like results when they are not: the
        # atlas sits at or below majority on valve and pump leakage.
        maj = R.get('R3_majority', {})
        for xi, c in enumerate(comps):
            m = maj.get(c)
            if m is None:
                continue
            ax.plot([xi - 2.4 * w, xi + 2.4 * w], [m * 100] * 2,
                    color=fs.C['black'], lw=1.4, ls=(0, (3, 2)), zorder=5)
        ax.plot([], [], color=fs.C['black'], lw=1.4, ls=(0, (3, 2)),
                label='majority class')
        ax.set_xticks(x); ax.set_xticklabels([COMP[c] for c in comps], fontsize=8.5)
        ax.set_ylabel('fault-state accuracy (%)'); ax.set_ylim(0, 118)
        ax.legend(fontsize=7, ncol=5, loc='upper center')
    ax.set_title('e   real fault diagnosis on the hydraulic rig, before and '
                 'after recalibration', loc='left', fontsize=9.5)

    fig.suptitle('Six real industrial systems: gas turbine, hydraulic rig, '
                 'motor bench, air compressor, transformer, wind farm',
                 y=0.955, fontsize=11)
    out = os.path.join(HERE, 'fig4_real.png')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.splitext(out)[0] + '.' + ext,
                    dpi=300 if ext == 'png' else None, format=ext)
        print('wrote', os.path.splitext(out)[0] + '.' + ext)
    print('wrote', out)


if __name__ == '__main__':
    main()
