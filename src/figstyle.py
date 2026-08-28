"""Shared plotting style: one visual language across every figure."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, colour-blind safe
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', sky='#56B4E9', yellow='#F0E442', grey='#5A5A5A',
         black='#111111')

ARM_COLOR = {
    'mono': C['grey'], 'raw': C['blue'], 'invariant': C['red'],
    'raw+inv': C['purple'], 'naive': C['green'], 'activity': C['orange'],
    'random': C['sky'], 'oracle': C['black'],
}
ARM_LABEL = {
    'mono': 'monolith', 'raw': 'raw state', 'invariant': 'invariants (R, Pe)',
    'raw+inv': 'raw + invariants', 'naive': 'Tr V, log det V',
    'activity': 'local activity', 'random': 'random proj.', 'oracle': 'oracle regime',
}
REGIME_COLOR = [C['sky'], C['blue'], C['orange'], C['red'], C['purple']]
REGIME_LABEL = ['slow, nominal', 'fast, nominal', 'slow, derated',
                'fast, derated', 'drive tripped']


def setup():
    mpl.rcParams.update({
        'figure.dpi': 130, 'savefig.dpi': 200,
        'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'axes.titlesize': 9.5, 'axes.labelsize': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'axes.grid': True,
        'grid.alpha': 0.18, 'grid.linewidth': 0.6,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5, 'legend.frameon': False,
        'lines.linewidth': 1.4, 'savefig.bbox': 'tight',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def panel_label(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10.5,
            fontweight='bold', va='top', ha='left')


def density_scatter(ax, x, y, c=None, cmap='viridis', s=1.5, alpha=0.35,
                    vlim=None, rasterized=True):
    if c is None:
        ax.scatter(x, y, s=s, alpha=alpha, c=C['blue'], lw=0, rasterized=rasterized)
        return None
    kw = {}
    if vlim is not None:
        kw = dict(vmin=vlim[0], vmax=vlim[1])
    return ax.scatter(x, y, s=s, alpha=alpha, c=c, cmap=cmap, lw=0,
                      rasterized=rasterized, **kw)


def robust_lim(v, lo=1.0, hi=99.0, pad=0.05):
    a, b = np.percentile(v, [lo, hi])
    m = (b - a) * pad
    return a - m, b + m


def sigfmt(v, n=3):
    return f'{v:.{n}g}'
