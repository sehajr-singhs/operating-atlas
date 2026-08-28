"""
Figures that show the object itself: the operating manifold, its axis
crossovers, the invariant fields defined on it, and the subsystem traces that
generate it.

These are the diagnostic figures for the paper's first half. Everything is
drawn from the cached prep artefacts, so nothing is refitted here.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

import figstyle as fs
from figstyle import C

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(ROOT, 'data')
FIGS = os.path.join(ROOT, 'figs')
os.makedirs(FIGS, exist_ok=True)

PMSM_AXES = ['u_q', 'u_d', 'i_q', 'i_d', 'motor_speed', 'torque', 'ambient', 'coolant']
PMSM_UNITS = ['V', 'V', 'A', 'A', 'rpm', 'N m', '$^\\circ$C', '$^\\circ$C']
PMSM_GROUP = ['electrical', 'electrical', 'electrical', 'electrical',
              'mechanical', 'mechanical', 'thermal', 'thermal']
GROUP_COLOR = dict(electrical=C['purple'], mechanical=C['blue'],
                   thermal=C['red'], geometric=C['green'])


def load(dataset='pmsm', k=3):
    blob = torch.load(os.path.join(DATA, f'prep_{dataset}_k{k}.pt'), weights_only=False)
    front = torch.load(os.path.join(DATA, f'prep_{dataset}_k{k}_front.pt'), weights_only=False)
    return blob, front


def _sub(n, m, seed=0):
    if n <= m:
        return np.arange(n)
    return np.random.RandomState(seed).choice(n, m, replace=False)


# ---------------------------------------------------------------------------
# 1. axis crossovers
# ---------------------------------------------------------------------------

def fig_axis_crossover(blob, front, split='train', n=40000, out='f_axes_pmsm.png'):
    """Pairwise projections of the operating axes, coloured by the drift-to-noise
    invariant. Where the axes cross is where a single global model has to serve
    two different physical regimes at the same point in one 2-d shadow."""
    fs.setup()
    S = blob['S'][split].numpy()
    idx = _sub(len(S), n)
    S = S[idx]
    Pe = front.invariants(blob['S'][split][idx]).numpy()[:, 1]
    cval = np.log1p(np.clip(Pe, 0, None))

    d = len(PMSM_AXES)
    fig, axes = plt.subplots(d, d, figsize=(13.5, 13.0))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(S[:, i], bins=60, color=GROUP_COLOR[PMSM_GROUP[i]],
                        alpha=0.75, lw=0)
                ax.set_yticks([])
            elif j < i:
                sc = fs.density_scatter(ax, S[:, j], S[:, i], c=cval,
                                        cmap='magma', s=0.8, alpha=0.30)
                ax.set_xlim(*fs.robust_lim(S[:, j]))
                ax.set_ylim(*fs.robust_lim(S[:, i]))
            else:
                r = np.corrcoef(S[:, j], S[:, i])[0, 1]
                ax.text(0.5, 0.5, f'{r:+.2f}', ha='center', va='center',
                        fontsize=11 + 7 * abs(r),
                        color=C['red'] if abs(r) > 0.5 else C['grey'],
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
            if i == d - 1 and j <= i:
                ax.set_xlabel(f'{PMSM_AXES[j]}\n({PMSM_UNITS[j]})', fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f'{PMSM_AXES[i]}\n({PMSM_UNITS[i]})', fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)

    handles = [plt.Line2D([], [], marker='s', ls='', ms=7, color=v, label=kk)
               for kk, v in GROUP_COLOR.items() if kk != 'geometric']
    fig.legend(handles=handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.965), title='subsystem')
    cb = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.015)
    cb.set_label('$\\log(1+\\mathrm{Pe})$  drift-to-noise invariant')
    fig.suptitle('PMSM operating axes: pairwise crossovers, coloured by the '
                 'routing invariant', y=0.985, fontsize=11)
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)
    print('  wrote', out)


# ---------------------------------------------------------------------------
# 2. the shape of the manifold
# ---------------------------------------------------------------------------

def fig_manifold_shape(blob, front, split='train', n=30000,
                       out='f_shape_pmsm.png', title='PMSM'):
    """The chart coordinates, i.e. the shape the operating point actually
    traces, viewed under three different colourings."""
    fs.setup()
    S = blob['S'][split]
    idx = _sub(len(S), n)
    Z = front.encode(S[idx]).numpy()
    inv = front.invariants(S[idx]).numpy()
    R, Pe = inv[:, 0], inv[:, 1]
    speed = np.r_[0, np.linalg.norm(np.diff(S[idx].numpy(), axis=0), axis=1)]

    fields = [(np.clip(R, *np.percentile(R, [1, 99])), 'scalar curvature $R$', 'coolwarm'),
              (np.log1p(np.clip(Pe, 0, None)), '$\\log(1+\\mathrm{Pe})$', 'magma'),
              (np.log1p(speed), 'log step size', 'viridis')]

    fig = plt.figure(figsize=(13.5, 4.4))
    for i, (v, lab, cm) in enumerate(fields):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        p = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=v, cmap=cm, s=1.2,
                       alpha=0.45, lw=0, rasterized=True)
        ax.set_xlabel('$z_1$', labelpad=-6); ax.set_ylabel('$z_2$', labelpad=-6)
        ax.set_zlabel('$z_3$', labelpad=-6)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_title(lab, pad=2)
        ax.view_init(elev=20, azim=35 + 8 * i)
        ax.grid(False)
        fig.colorbar(p, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle(f'{title}: the operating manifold in chart coordinates',
                 y=1.01, fontsize=11)
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)
    print('  wrote', out)


# ---------------------------------------------------------------------------
# 3. invariant fields over the chart
# ---------------------------------------------------------------------------

def fig_invariant_fields(blob, front, split='train', n=60000,
                         out='f_fields_pmsm.png', title='PMSM'):
    """R and Pe as fields on 2-d slices of the chart, plus their joint
    distribution. This is the routing coordinate system itself."""
    fs.setup()
    S = blob['S'][split]
    idx = _sub(len(S), n)
    Z = front.encode(S[idx]).numpy()
    inv = front.invariants(S[idx]).numpy()
    R, Pe = inv[:, 0], inv[:, 1]
    lPe = np.log1p(np.clip(Pe, 0, None))
    Rc = np.clip(R, *np.percentile(R, [1, 99]))

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for col, (a, b) in enumerate(pairs):
        for row, (v, lab, cm) in enumerate(
                [(Rc, 'scalar curvature $R$', 'coolwarm'),
                 (lPe, '$\\log(1+\\mathrm{Pe})$', 'magma')]):
            ax = axes[row, col]
            sc = fs.density_scatter(ax, Z[:, a], Z[:, b], c=v, cmap=cm,
                                    s=1.4, alpha=0.40)
            ax.set_xlabel(f'$z_{a+1}$'); ax.set_ylabel(f'$z_{b+1}$')
            if col == 2:
                fig.colorbar(sc, ax=ax, fraction=0.045).set_label(lab, fontsize=7)
            if row == 0 and col == 0:
                fs.panel_label(ax, 'a')
            if row == 1 and col == 0:
                fs.panel_label(ax, 'b')
    fig.suptitle(f'{title}: the two routing invariants as fields on the chart',
                 y=0.99, fontsize=11)
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)

    # joint distribution: are they independent coordinates or one dressed twice?
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    h = ax[0].hexbin(Rc, lPe, gridsize=55, cmap='inferno', bins='log', mincnt=1)
    ax[0].set_xlabel('scalar curvature $R$')
    ax[0].set_ylabel('$\\log(1+\\mathrm{Pe})$')
    fig.colorbar(h, ax=ax[0], fraction=0.045).set_label('count', fontsize=7)
    rho = np.corrcoef(np.argsort(np.argsort(Rc)), np.argsort(np.argsort(lPe)))[0, 1]
    ax[0].set_title(f'joint density  (Spearman {rho:+.2f})')
    ax[1].hist(lPe, bins=80, color=C['red'], alpha=0.8, lw=0)
    ax[1].set_xlabel('$\\log(1+\\mathrm{Pe})$'); ax[1].set_ylabel('count')
    ax[1].set_title('advective vs diffusive operating points')
    ax[1].axvline(np.log1p(1.0), color=C['black'], ls='--', lw=1.0)
    ax[1].text(np.log1p(1.0), ax[1].get_ylim()[1] * 0.92, '  Pe = 1',
               fontsize=7, va='top')
    fig.suptitle(f'{title}: are the invariants two coordinates or one?',
                 y=1.02, fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out.replace('fields', 'jointinv')))
    plt.close(fig)
    print('  wrote', out, 'and joint')


# ---------------------------------------------------------------------------
# 4. subsystem traces
# ---------------------------------------------------------------------------

def fig_subsystems(blob, front, split='test', out='f_subsys_pmsm.png'):
    """One measurement session, split by subsystem, with the invariants beneath.
    Shows what the operating point is doing when the routing coordinates move."""
    fs.setup()
    G = blob['G'][split].numpy()
    sess = np.bincount(G).argmax()
    m = G == sess
    S = blob['S'][split][m]
    Y = blob['Yn'][split][m].numpy() * blob['ys'].numpy()
    inv = front.invariants(S).numpy()
    Sn = S.numpy()
    t = np.arange(len(Sn)) * 0.5 * 4 / 60.0          # minutes (2 Hz, stride 4)

    groups = [('electrical', ['u_q', 'u_d', 'i_q', 'i_d']),
              ('mechanical', ['motor_speed', 'torque']),
              ('thermal', ['ambient', 'coolant'])]
    fig, axes = plt.subplots(5, 1, figsize=(9.5, 9.5), sharex=True)
    for ax, (gname, cols) in zip(axes[:3], groups):
        for c in cols:
            j = PMSM_AXES.index(c)
            ax.plot(t, Sn[:, j], lw=0.9, label=f'{c} ({PMSM_UNITS[j]})')
        ax.set_ylabel(gname, color=GROUP_COLOR[gname], fontweight='bold')
        ax.legend(ncol=4, loc='upper right')

    ax = axes[3]
    for i, nm in enumerate(blob['targets']):
        ax.plot(t, Y[:, i], lw=1.0, label=nm)
    ax.set_ylabel('internal\ntemps ($^\\circ$C)')
    ax.legend(ncol=4, loc='upper right')

    ax = axes[4]
    ax.plot(t, np.log1p(np.clip(inv[:, 1], 0, None)), color=C['red'], lw=1.0,
            label='$\\log(1+\\mathrm{Pe})$')
    ax2 = ax.twinx()
    ax2.plot(t, inv[:, 0], color=C['blue'], lw=0.9, alpha=0.8, label='$R$')
    ax2.set_ylabel('$R$', color=C['blue']); ax2.grid(False)
    ax.set_ylabel('$\\log(1+\\mathrm{Pe})$', color=C['red'])
    ax.set_xlabel('time (min)')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    fig.suptitle(f'PMSM session {sess}: subsystems and the routing invariants',
                 y=0.995, fontsize=11)
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    print('loading pmsm ...')
    blob, front = load('pmsm', 3)
    fig_manifold_shape(blob, front)
    fig_invariant_fields(blob, front)
    fig_subsystems(blob, front)
    fig_axis_crossover(blob, front)
    print('done')
