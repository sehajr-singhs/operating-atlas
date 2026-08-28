"""
Figure 5 -- the controlled shape-vs-level fleet experiment, three platforms,
plus the deep learned baseline head-to-head.

Three fleets of 48 MuJoCo machines each (UR5e, Panda, iiwa14), identical in
everything except which physical parameters carry the identity:

    level_only   payload varies, couplings neutral
    shape_only   payload fixed, six relational parameters vary
    all          everything varies

Row 1: top-1 sibling retrieval by identity axis (atlas vs marginals vs
calibration block), binomial stars against chance.
Row 2: the deep learned baseline (autoencoder embedding, trained on raw
telemetry windows) on the same protocol, before and after an independent
per-channel recalibration of the held-out half -- the learned representation
wins on clean data and collapses under re-instrumentation, while the atlas is
numerically unmoved.

Reads results.json from the ioo-shape-level{,panda,iiwa14} Kaggle kernels.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import figstyle as fs

PLATFORM_RES = {
    'UR5e': os.path.join(os.path.expanduser('~'), 'kaggle_kernel',
                         'shape_level2_out', 'results.json'),
    'Panda': os.path.join(os.path.expanduser('~'), 'kaggle_kernel',
                          'shape_level_p_out', 'results.json'),
    'iiwa14': os.path.join(os.path.expanduser('~'), 'kaggle_kernel',
                           'shape_level_i_out', 'results.json'),
}

ORDER = ['level_only', 'shape_only', 'all']
LABEL = {'level_only': 'level-only',
         'shape_only': 'shape-only',
         'all': 'both axes'}
C = fs.C


def load_platforms():
    out = {}
    for plat, path in PLATFORM_RES.items():
        if os.path.exists(path):
            out[plat] = json.load(open(path))['per_fleet']
    return out


def main():
    fs.setup()
    plats = load_platforms()
    names = list(plats)
    chance = 100.0 / plats[names[0]][ORDER[0]]['P2_atlas']['n']

    fig = plt.figure(figsize=(12.2, 7.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
                  left=0.075, right=0.975, top=0.88, bottom=0.10)

    for j, plat in enumerate(names):
        pf = plats[plat]
        x = np.arange(len(ORDER))
        w = 0.24

        # ---- row 1: axis decomposition -----------------------------------
        ax = fig.add_subplot(gs[0, j])
        for i, (key, color, lab) in enumerate([
                ('P2_atlas', C['blue'], 'atlas (8 invariant atoms)'),
                ('P2_marginal', C['grey'], 'per-channel marginals'),
                ('P2_calibration', C['orange'], 'calibration block')]):
            vals = [100 * pf[n][key]['top1'] for n in ORDER]
            ps = [pf[n][key].get('p_binom') for n in ORDER]
            bars = ax.bar(x + (i - 1) * w, vals, w, color=color, alpha=0.92,
                          label=lab if j == 0 else None)
            for b, p in zip(bars, ps):
                if p is None:
                    continue
                star = '***' if p < 0.001 else ('**' if p < 0.01 else
                                                ('*' if p < 0.05 else 'ns'))
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                        star, ha='center', va='bottom', fontsize=6.5)
        ax.axhline(chance, color=C['black'], ls='--', lw=1.0)
        ax.text(2.42, chance + 0.7, f'chance {chance:.1f}%', fontsize=6.8,
                ha='right')
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[n] for n in ORDER], fontsize=8)
        if j == 0:
            ax.set_ylabel('sibling retrieval, top-1 (%)')
        ax.set_ylim(0, 62)
        if j == 0:
            ax.legend(fontsize=6.6, loc='upper left', frameon=False)
        ax.set_title(f"{'abcd'[j]}   {plat}: retrieval by identity axis",
                     loc='left', fontsize=9.5)

        # ---- row 2: deep baseline collapse --------------------------------
        ax = fig.add_subplot(gs[1, j])
        for i, n in enumerate(ORDER):
            p8 = pf[n]['P8']
            ae_clean = 100 * p8['ae_clean']['top1']
            ae_warp = 100 * p8['ae_warped']['top1']
            atl = 100 * pf[n]['P2_atlas']['top1']
            atl_w = 100 * pf[n]['P6_atlas']['top1']
            ax.bar(x[i] - 0.22, ae_clean, 0.18, color=C['green'], alpha=0.92,
                   label='AE clean' if (i == 0 and j == 0) else None)
            ax.bar(x[i] - 0.00, ae_warp, 0.18, color=C['green'], alpha=0.35,
                   edgecolor=C['green'], hatch='//',
                   label='AE after recalibration' if (i == 0 and j == 0)
                   else None)
            ax.bar(x[i] + 0.22, atl, 0.18, color=C['blue'], alpha=0.92,
                   label='atlas' if (i == 0 and j == 0) else None)
            ax.plot([x[i] + 0.22 - 0.09, x[i] + 0.22 + 0.09],
                    [atl_w, atl_w], color=C['red'], lw=1.6,
                    label='atlas after recalibration'
                    if (i == 0 and j == 0) else None)
            if ae_clean > 1 and ae_warp / max(ae_clean, 1e-9) < 0.25:
                ax.annotate(f'{ae_clean:.0f}\u2192{ae_warp:.0f}',
                            (x[i], ae_clean), textcoords='offset points',
                            xytext=(0, 4), ha='center', fontsize=6.5,
                            color=C['green'])
        ax.axhline(chance, color=C['black'], ls=':', lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[n] for n in ORDER], fontsize=8)
        if j == 0:
            ax.set_ylabel('top-1 retrieval (%)')
        ax.set_ylim(0, 66)
        if j == 0:
            ax.legend(fontsize=6.2, loc='upper left', frameon=False)
        ax.set_title(f"{'def'[j]}   {plat}: learned vs invariant under\n"
                     're-instrumentation', loc='left', fontsize=9.5)

    fig.suptitle('Machine identity has two axes: absolute level (payload) and '
                 'relational physics (couplings). The invariant atlas is blind '
                 'to one and lives on the other; a learned representation '
                 'outreads it on clean data and is destroyed by sensor change.',
                 y=0.965, fontsize=10.5)
    for ext in ('png', 'pdf'):
        out = os.path.join(HERE, f'fig5_fleet_control.{ext}')
        fig.savefig(out, dpi=300 if ext == 'png' else None, format=ext)
    print('wrote fig5_fleet_control.png + .pdf')


if __name__ == '__main__':
    main()
