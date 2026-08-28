"""Figures for the validation and results sections."""

import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
from figstyle import C, ARM_COLOR, ARM_LABEL

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
FIGS = os.path.join(ROOT, 'figs')
os.makedirs(FIGS, exist_ok=True)
ARMS = ['mono', 'raw', 'invariant', 'raw+inv', 'naive', 'activity', 'random']


def _load(p):
    f = os.path.join(RES, p)
    return json.load(open(f)) if os.path.exists(f) else None


# ---------------------------------------------------------------------------

def fig_cost_and_theory(out='f_validation.png'):
    """Curvature cost against chart dimension, and the Theorem 1 residual."""
    fs.setup()
    k = [2, 3, 4, 5, 6, 8]
    ms = [1.21, 1.98, 3.63, 12.67, 24.36, 115.21]
    taus = [0.25, 0.5, 1.0, 2.0, 4.0]
    err = [2.082e-17, 8.674e-18, 5.204e-18, 1.952e-18, 1.084e-18]
    sens = [0.0149, 0.0074, 0.0037, 0.0018, 0.0009]

    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.4))
    ax[0].semilogy(k, ms, 'o-', color=C['blue'])
    ax[0].set_xlabel('chart dimension $k$')
    ax[0].set_ylabel('ms per point')
    ax[0].set_title('cost of the exact invariants')
    ax[0].annotate('$115$ ms', xy=(8, 115), xytext=(5.6, 60), fontsize=7.5,
                   arrowprops=dict(arrowstyle='->', lw=0.7))
    fs.panel_label(ax[0], 'a')

    ax[1].loglog(taus, sens, 'o-', color=C['red'], label='$\\|\\partial F/\\partial r\\|$')
    ax[1].loglog(taus, [sens[2] * 1.0 / t for t in taus], '--', color=C['grey'],
                 label='$\\propto 1/\\tau$')
    ax[1].set_xlabel('gate temperature $\\tau$')
    ax[1].set_ylabel('mean sensitivity')
    ax[1].set_title('Theorem 1: predicted scaling')
    ax[1].legend()
    fs.panel_label(ax[1], 'b')

    ax[2].semilogy(taus, err, 'o-', color=C['green'])
    ax[2].axhline(2.2e-16, color=C['grey'], ls=':', lw=1.0)
    ax[2].text(0.3, 2.6e-16, 'double precision $\\epsilon$', fontsize=7,
               color=C['grey'])
    ax[2].set_xlabel('gate temperature $\\tau$')
    ax[2].set_ylabel('max identity residual')
    ax[2].set_title('Theorem 1: exactness')
    ax[2].set_ylim(1e-19, 1e-14)
    fs.panel_label(ax[2], 'c')
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)
    print('  wrote', out)


def fig_invariance(out='f_invariance.png'):
    """How well each candidate coordinate survives a coordinate change, both
    analytically and after estimation from finite samples."""
    fs.setup()
    names = [r'$\mathrm{Pe}$', r'$R$', '$\\mathrm{Tr}\\,V$' + '\n(control)']
    zw = [0.999, 0.604, 0.972]
    zw_sd = [0.000, 0.060, 0.0]
    truth = [0.998, 0.716, np.nan]

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    x = np.arange(len(names))
    ax[0].bar(x - 0.18, zw, 0.34, yerr=zw_sd, color=[C['red'], C['blue'], C['grey']],
              capsize=3, label='across coordinate change')
    ax[0].bar(x + 0.18, truth, 0.34, color=[C['red'], C['blue'], C['grey']],
              alpha=0.45, label='against ground truth')
    ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("Spearman $\\rho$"); ax[0].set_ylim(0, 1.05)
    ax[0].axhline(1.0, color=C['black'], ls=':', lw=0.8)
    ax[0].set_title('invariance after estimation')
    ax[0].legend(loc='lower left')
    fs.panel_label(ax[0], 'a')

    ax[1].bar([0, 1], [1.88e-8, 9.66e-7], 0.5, color=[C['red'], C['blue']])
    ax[1].set_yscale('log')
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['$\\mathrm{Pe}$', '$R$'])
    ax[1].set_ylabel('relative error')
    ax[1].set_title('analytic push-forward (exact algebra)')
    ax[1].axhline(2.2e-16, color=C['grey'], ls=':', lw=0.8)
    fs.panel_label(ax[1], 'b')
    fig.suptitle('The two invariants are exactly invariant on paper; only one '
                 'survives estimation', y=1.03, fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)
    print('  wrote', out)


def fig_main_results(datasets=('cmapss', 'pmsm', 'ur5e'), out='f_results.png'):
    """Test error by routing arm, normalised to the raw-state router."""
    fs.setup()
    summ = _load('summary.json')
    if not summ:
        print('  no summary.json yet'); return
    ds = [d for d in datasets if d in summ]
    if not ds:
        print('  no datasets in summary'); return
    fig, axes = plt.subplots(1, len(ds), figsize=(4.3 * len(ds), 3.8))
    for ax, d in zip(np.atleast_1d(axes), ds):
        arms = summ[d]['arms']
        ref = arms.get('raw', {}).get('test_mean', np.nan)
        present = [a for a in ARMS if a in arms]
        vals = [100 * (arms[a]['test_mean'] - ref) / ref for a in present]
        errs = [100 * arms[a]['test_sd'] / ref for a in present]
        ax.barh(range(len(present)), vals, xerr=errs,
                color=[ARM_COLOR[a] for a in present], capsize=3)
        ax.axvline(0, color=C['black'], lw=1.0)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([ARM_LABEL[a] for a in present])
        ax.invert_yaxis()
        ax.set_xlabel('test error vs raw-state router (%)')
        n = arms[present[0]]['n']
        ax.set_title(f'{d}   (n={n} seeds)')
    fig.suptitle('In-distribution: no routing coordinate separates from the '
                 'raw state', y=1.02, fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)
    print('  wrote', out)


def fig_regime(platform='ur5e', out=None):
    """Regime discovery at matched dimension."""
    fs.setup()
    r = _load(f'regime_{platform}.json')
    if not r:
        print(f'  no regime_{platform}.json yet'); return
    names = list(r['scores'])
    ami = [r['scores'][n]['ami'][0] for n in names]
    amis = [r['scores'][n]['ami'][1] for n in names]
    bac = [r['scores'][n]['bacc'][0] for n in names]
    bacs = [r['scores'][n]['bacc'][1] for n in names]
    order = np.argsort(ami)[::-1]
    names = [names[i] for i in order]
    ami = [ami[i] for i in order]; amis = [amis[i] for i in order]
    bac = [bac[i] for i in order]; bacs = [bacs[i] for i in order]
    col = [C['grey'] if 'reference' in n else
           (C['red'] if n == 'invariant' else C['blue']) for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax[0].barh(range(len(names)), ami, xerr=amis, color=col, capsize=3)
    ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names)
    ax[0].invert_yaxis(); ax[0].set_xlabel('adjusted mutual information')
    ax[0].set_title('regime information in the coordinates')
    ax[1].barh(range(len(names)), bac, xerr=bacs, color=col, capsize=3)
    ax[1].axvline(0.2, color=C['black'], ls='--', lw=0.9)
    ax[1].text(0.2, len(names) - 0.4, ' chance', fontsize=7)
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels([])
    ax[1].invert_yaxis(); ax[1].set_xlabel('balanced accuracy (5 regimes)')
    ax[1].set_title('k-NN regime recovery')
    fig.suptitle(f'{platform.upper()}: how much of the physics each 2-d '
                 'coordinate pair carries', y=1.03, fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out or f'f_regime_{platform}.png'))
    plt.close(fig)
    print('  wrote', out or f'f_regime_{platform}.png')


def fig_transfer(out='f_transfer.png'):
    """Degradation under sensor recalibration, with and without an
    accompanying shift in the operating distribution."""
    fs.setup()
    r = _load('pmsm_transfer.json')
    if not r:
        print('  no pmsm_transfer.json yet'); return
    rows = r['transfer']
    clean = r['clean']
    modes = ['channelwise', 'mixing']
    dists = ['matched', 'shifted']
    defence = 'quantile'
    fig, axes = plt.subplots(len(dists), len(modes),
                             figsize=(5.0 * len(modes), 3.5 * len(dists)),
                             sharey=True)
    axes = np.atleast_2d(axes)
    for i, dist in enumerate(dists):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            sel = [x for x in rows if x['dist'] == dist and x['mode'] == mode
                   and x['defence'] == defence]
            if not sel:
                ax.text(0.5, 0.5, 'no runs', ha='center', transform=ax.transAxes)
                continue
            present = [a for a in ARMS if a in sel[0]['arms']]
            vals, errs = [], []
            for a in present:
                v = [x['arms'][a]['mean'] / clean[a]['mean'] for x in sel]
                vals.append(np.mean(v)); errs.append(np.std(v))
            ax.bar(range(len(present)), vals, yerr=errs,
                   color=[ARM_COLOR[a] for a in present], capsize=3)
            ax.axhline(1.0, color=C['black'], ls='--', lw=0.9)
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels([ARM_LABEL[a] for a in present], rotation=35,
                               ha='right', fontsize=7)
            ax.set_title(f'{dist} distribution, {mode} warp')
            if j == 0:
                ax.set_ylabel('test error / clean error')
    fig.suptitle('Transfer under sensor recalibration (quantile-matched '
                 'defence; lower is better)', y=1.0, fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    fig_cost_and_theory()
    fig_invariance()
    fig_main_results()
    fig_regime('ur5e')
    fig_transfer()
