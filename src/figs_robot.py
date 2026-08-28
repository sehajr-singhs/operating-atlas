"""
Robot figures: per-joint sub-axes, the coupled subsystems, and the operating
manifold coloured by the ground-truth regime the model never sees.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

import figstyle as fs
from figstyle import C, REGIME_COLOR, REGIME_LABEL
from robot_sim import PLATFORMS, T_KNEE, T_TRIP, rollout

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(ROOT, 'data')
FIGS = os.path.join(ROOT, 'figs')
os.makedirs(FIGS, exist_ok=True)


def load_npz(platform):
    d = np.load(os.path.join(DATA, f'robot_{platform}.npz'))
    return d['X'], d['Y'], d['S'], d['G'], d['L']


def split_state(S, nj):
    return dict(q=S[:, :nj], dq=S[:, nj:2 * nj], tau=S[:, 2 * nj:3 * nj],
                Tw=S[:, 3 * nj:4 * nj], Th=S[:, 4 * nj:5 * nj],
                Tenv=S[:, 5 * nj], payload=S[:, 5 * nj + 1])


# ---------------------------------------------------------------------------

def fig_joint_axes(platform='ur5e', n=25000, out=None):
    """Per-joint phase portraits, one sub-axis per joint, coloured by winding
    temperature. The crossover between the mechanical and thermal axes is the
    whole point: the same (q, dq) point sits in different physics depending on
    how hot the joint is."""
    fs.setup()
    S, L = load_npz(platform)[2], load_npz(platform)[4]
    nj = PLATFORMS[platform]['n']
    idx = np.random.RandomState(0).choice(len(S), min(n, len(S)), replace=False)
    P = split_state(S[idx], nj)
    Lb = L[idx]

    fig, axes = plt.subplots(3, nj, figsize=(2.05 * nj, 6.6))
    for j in range(nj):
        ax = axes[0, j]
        sc = ax.scatter(P['q'][:, j], P['dq'][:, j], c=P['Tw'][:, j], s=1.0,
                        cmap='inferno', alpha=0.35, lw=0, rasterized=True,
                        vmin=25, vmax=T_TRIP)
        ax.set_title(f'joint {j}', fontsize=8.5)
        if j == 0:
            ax.set_ylabel('$\\dot q$  (rad/s)')
        ax.set_xlabel('$q$ (rad)', fontsize=7)

        ax = axes[1, j]
        ax.scatter(P['dq'][:, j], P['tau'][:, j], c=P['Tw'][:, j], s=1.0,
                   cmap='inferno', alpha=0.35, lw=0, rasterized=True,
                   vmin=25, vmax=T_TRIP)
        tm = PLATFORMS[platform]['tau_max'][j]
        ax.axhline(tm, color=C['grey'], ls=':', lw=0.8)
        ax.axhline(-tm, color=C['grey'], ls=':', lw=0.8)
        if j == 0:
            ax.set_ylabel('$\\tau$  (N m)')
        ax.set_xlabel('$\\dot q$', fontsize=7)

        ax = axes[2, j]
        ax.scatter(P['tau'][:, j] ** 2, P['Tw'][:, j], c=P['Th'][:, j], s=1.0,
                   cmap='viridis', alpha=0.35, lw=0, rasterized=True)
        ax.axhline(T_KNEE, color=C['orange'], ls='--', lw=0.9)
        ax.axhline(T_TRIP, color=C['red'], ls='--', lw=0.9)
        if j == 0:
            ax.set_ylabel('$T_w$  ($^\\circ$C)')
        ax.set_xlabel('$\\tau^2$', fontsize=7)
        ax.set_xscale('symlog')
    for ax in axes.ravel():
        ax.tick_params(labelsize=6)
    cb = fig.colorbar(sc, ax=axes[:2, :], fraction=0.012, pad=0.01)
    cb.set_label('winding temperature $T_w$ ($^\\circ$C)', fontsize=7)
    fig.suptitle(f'{platform.upper()}: per-joint sub-axes. '
                 'top mechanical phase portrait, middle torque-speed envelope, '
                 'bottom ohmic heating law', y=0.995, fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out or f'f_axes_{platform}.png'))
    plt.close(fig)
    print('  wrote', out or f'f_axes_{platform}.png')


def fig_subsystems(platform='ur5e', seed=3, seconds=180.0, out=None):
    """One episode, subsystem by subsystem, with the regime strip beneath."""
    fs.setup()
    A, lab, names = rollout(platform, seconds=seconds, seed=seed)
    nj = PLATFORMS[platform]['n']
    P = split_state(A, nj)
    t = np.arange(len(A)) / 50.0

    fig, axes = plt.subplots(5, 1, figsize=(10.0, 9.8), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 1, 0.28]))
    cmapj = plt.cm.viridis(np.linspace(0.1, 0.9, nj))
    for j in range(nj):
        axes[0].plot(t, P['q'][:, j], lw=0.7, color=cmapj[j], label=f'j{j}')
        axes[1].plot(t, P['dq'][:, j], lw=0.7, color=cmapj[j])
        axes[2].plot(t, P['tau'][:, j], lw=0.7, color=cmapj[j])
        axes[3].plot(t, P['Tw'][:, j], lw=0.9, color=cmapj[j])
        axes[3].plot(t, P['Th'][:, j], lw=0.7, color=cmapj[j], ls=':', alpha=0.7)
    axes[0].set_ylabel('$q$ (rad)\nkinematic')
    axes[1].set_ylabel('$\\dot q$ (rad/s)\nkinematic')
    axes[2].set_ylabel('$\\tau$ (N m)\nmechanical')
    axes[3].set_ylabel('$T_w,\\,T_h$ ($^\\circ$C)\nthermal')
    axes[3].axhline(T_KNEE, color=C['orange'], ls='--', lw=1.0)
    axes[3].axhline(T_TRIP, color=C['red'], ls='--', lw=1.0)
    axes[3].text(t[-1], T_KNEE, ' derate', color=C['orange'], fontsize=7, va='bottom', ha='right')
    axes[3].text(t[-1], T_TRIP, ' trip', color=C['red'], fontsize=7, va='bottom', ha='right')
    axes[0].legend(ncol=nj, loc='upper right', fontsize=6.5)

    ax = axes[4]
    ax.imshow(lab[None, :], aspect='auto', interpolation='nearest',
              extent=[t[0], t[-1], 0, 1], cmap=plt.matplotlib.colors.ListedColormap(REGIME_COLOR),
              vmin=-0.5, vmax=4.5)
    ax.set_yticks([]); ax.set_xlabel('time (s)'); ax.set_ylabel('regime')
    ax.grid(False)
    present = sorted(set(lab.tolist()))
    ax.legend(handles=[Line2D([], [], marker='s', ls='', ms=7,
                              color=REGIME_COLOR[i], label=REGIME_LABEL[i])
                       for i in present],
              ncol=len(present), loc='upper center', bbox_to_anchor=(0.5, -0.75),
              fontsize=7)
    fig.suptitle(f'{platform.upper()} episode {seed}: the coupled subsystems, '
                 'and the regime they put the machine in', y=0.995, fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out or f'f_subsys_{platform}.png'))
    plt.close(fig)
    print('  wrote', out or f'f_subsys_{platform}.png')


def fig_thermal_coupling(platforms=('ur5e', 'panda', 'iiwa14'), out='f_thermal_platforms.png'):
    """Thermal headroom across platforms. The iiwa14 is oversized for this duty
    and never leaves the nominal regime; that contrast is physical, not a
    tuning artefact, and it is what the routing has to cope with."""
    fs.setup()
    fig, axes = plt.subplots(1, len(platforms), figsize=(4.0 * len(platforms), 3.4),
                             sharey=True)
    for ax, p in zip(np.atleast_1d(axes), platforms):
        f = os.path.join(DATA, f'robot_{p}.npz')
        if not os.path.exists(f):
            ax.text(0.5, 0.5, f'{p}\nnot generated', ha='center', va='center',
                    transform=ax.transAxes); continue
        S, L = load_npz(p)[2], load_npz(p)[4]
        nj = PLATFORMS[p]['n']
        P = split_state(S, nj)
        tau_n = np.abs(P['tau']) / np.asarray(PLATFORMS[p]['tau_max'])
        idx = np.random.RandomState(0).choice(len(S), min(40000, len(S)), replace=False)
        sc = ax.scatter(tau_n[idx].mean(1), P['Tw'][idx].max(1),
                        c=[REGIME_COLOR[i] for i in L[idx]], s=1.2, alpha=0.35,
                        lw=0, rasterized=True)
        ax.axhline(T_KNEE, color=C['orange'], ls='--', lw=1.0)
        ax.axhline(T_TRIP, color=C['red'], ls='--', lw=1.0)
        frac = float((L[idx] >= 2).mean())
        ax.set_title(f'{p.upper()}   {100*frac:.0f}% derated or tripped')
        ax.set_xlabel('mean $|\\tau| / \\tau_{\\max}$')
    np.atleast_1d(axes)[0].set_ylabel('max winding $T_w$ ($^\\circ$C)')
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=5,
                               color=REGIME_COLOR[i], label=REGIME_LABEL[i])
                        for i in range(5)],
               ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.06), fontsize=7.5)
    fig.suptitle('Thermal headroom differs by platform under the same excitation',
                 y=1.02, fontsize=10.5)
    fig.savefig(os.path.join(FIGS, out))
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    import sys
    plats = sys.argv[1:] or ['ur5e']
    for p in plats:
        fig_joint_axes(p)
        fig_subsystems(p)
    fig_thermal_coupling(tuple(plats) if len(plats) > 1 else ('ur5e',))
