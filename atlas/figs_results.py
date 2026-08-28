"""
Figure 2 -- what the atlas buys, what it needs, and what it costs.

Four panels, four different jobs, so four different forms:
  a  telemetry budget          change over a quantity      -> lines
  b  identification by feature set   magnitude comparison  -> bars
  c  identification by atom          magnitude comparison  -> bars
  d  recalibration                   before/after identity -> paired bars
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

# three highlighted atoms, greys for the rest: nine coloured series would be
# unreadable and would need hues past any validated categorical set
HI = {'levy': fs.C['red'], 'tau': fs.C['blue'], 'fill': fs.C['green']}


def main():
    fs.setup()
    R = json.load(open(os.path.join(HERE, 'results_pmsm.json')))
    u = R['n_units']
    chance = 100.0 / u

    fig = plt.figure(figsize=(11.4, 7.4))
    gs = GridSpec(2, 12, figure=fig, hspace=0.52, wspace=2.4,
                  left=0.07, right=0.985, top=0.90, bottom=0.10)

    # --- a telemetry budget ---------------------------------------------
    ax = fig.add_subplot(gs[0, :6])
    NS = R['budget']['N']
    for nm in atoms.ATOM_NAMES:
        v = R['budget']['disjoint'][nm]
        c = HI.get(nm, fs.C['grey'])
        ax.plot(NS, v, color=c, lw=2.0 if nm in HI else 0.9,
                alpha=1.0 if nm in HI else 0.35, zorder=3 if nm in HI else 1)
        if nm in HI:
            ax.annotate(nm, xy=(NS[-1], v[-1]), xytext=(6, 0),
                        textcoords='offset points', color=c, fontsize=8,
                        va='center')
    mi = np.nanmean([R['budget']['interleaved'][nm] for nm in atoms.ATOM_NAMES], 0)
    ax.plot(NS, mi, color=fs.C['black'], lw=1.4, ls='--', zorder=4)
    ax.annotate('interleaved split\n(estimation noise only)',
                xy=(NS[len(NS) // 2], mi[len(NS) // 2]), xytext=(0, -30),
                textcoords='offset points', fontsize=7.5, ha='center',
                color=fs.C['black'])
    ax.set_xscale('log')
    ax.set_xlabel('samples per unit  $N$')
    ax.set_ylabel('atlas reproducibility (r)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('a   atoms need $10^3$ to $10^4$ samples; the residual gap is '
                 'drift, not noise', loc='left', fontsize=9.5)
    ax.axvspan(100, 400, color=fs.C['red'], alpha=0.07, zorder=0)
    ax.text(190, 0.16, 'C-MAPSS\nlives here', fontsize=7, ha='center',
            color=fs.C['red'])

    # --- b identification by feature set ---------------------------------
    ax = fig.add_subplot(gs[0, 6:])
    lab = {'atlas, invariant (8)': 'atlas, invariant 8',
           'full atlas (9)': 'full atlas (9 atoms)',
           'marginals': 'per-channel marginals',
           'rho only': 'correlation matrix\n(rho alone)'}
    col = {'atlas, invariant (8)': fs.C['orange'], 'full atlas (9)': fs.C['red'],
           'marginals': fs.C['grey'], 'rho only': fs.C['blue']}
    order = sorted(lab, key=lambda k: R['identification'][k]['top1'])
    v = [R['identification'][k]['top1'] for k in order]
    y = np.arange(len(order))
    ax.barh(y, v, color=[col[k] for k in order], height=0.62)
    for yy, vv in zip(y, v):
        ax.text(vv + 0.8, yy, f'{vv:.1f}%', va='center', fontsize=8)
    ax.axvline(chance, color=fs.C['black'], lw=1.0, ls=':')
    ax.text(chance + 0.5, -0.62, f'chance {chance:.1f}%', fontsize=7,
            color=fs.C['black'], va='center')
    ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in order], fontsize=8)
    ax.set_ylim(-1.0, len(order) - 0.4)
    ax.set_xlabel('top-1 identification of 40 real motor sessions (%)')
    ax.set_xlim(0, max(v) * 1.30)
    ax.set_title('b   fourfold over a correlation matrix, but only slightly\n'
                 '      over marginals, which panel d then destroys',
                 loc='left', fontsize=9.5)

    # --- c per atom -------------------------------------------------------
    ax = fig.add_subplot(gs[1, :7])
    nms = sorted(atoms.ATOM_NAMES, key=lambda n: -R['per_atom'][n]['top1'])
    v = [R['per_atom'][n]['top1'] for n in nms]
    cols = [HI.get(n, fs.C['grey']) for n in nms]
    ax.bar(np.arange(len(nms)), v, color=cols, width=0.68)
    ax.axhline(chance, color=fs.C['black'], lw=1.0, ls=':')
    ax.text(len(nms) - 0.4, chance + 0.6, f'chance {chance:.1f}%', fontsize=7,
            ha='right')
    ax.set_xticks(np.arange(len(nms)))
    ax.set_xticklabels(nms, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('top-1 (%)')
    ax.set_title('c   oriented hysteresis carries most of the identity, and it '
                 'is exactly\n      what a symmetric correlation cannot express',
                 loc='left', fontsize=9.5)

    # --- d recalibration --------------------------------------------------
    ax = fig.add_subplot(gs[1, 7:])
    before = [R['identification']['atlas, invariant (8)']['top1'],
              R['identification']['marginals']['top1']]
    after = [R['invariance']['atlas_warped']['top1'],
             R['invariance']['marginal_warped']['top1']]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, before, width=w, color=fs.C['red'], label='as calibrated')
    ax.bar(x + w / 2, after, width=w, color=fs.C['sky'],
           label='after recalibration')
    for xx, b, a in zip(x, before, after):
        ax.text(xx - w / 2, b + 0.6, f'{b:.1f}', ha='center', fontsize=7.5)
        ax.text(xx + w / 2, a + 0.6, f'{a:.1f}', ha='center', fontsize=7.5)
    ax.axhline(chance, color=fs.C['black'], lw=1.0, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(['atlas\n(invariant 8)', 'marginals'],
                                         fontsize=8)
    ax.set_ylabel('top-1 (%)')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_title('d   an independent monotone warp\n      on every channel',
                 loc='left', fontsize=9.5)

    fig.suptitle('PMSM test bench, 40 independent sessions: identification, '
                 'sample budget, and recalibration', y=0.965, fontsize=11)
    out = os.path.join(HERE, 'fig2_results.png')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.splitext(out)[0] + '.' + ext,
                    dpi=300 if ext == 'png' else None, format=ext)
        print('wrote', os.path.splitext(out)[0] + '.' + ext)
    print('wrote', out)
    print('  shift per atom:', {k: f'{v:.1e}' for k, v
                                in R['invariance']['atom_shift'].items()})


if __name__ == '__main__':
    main()
