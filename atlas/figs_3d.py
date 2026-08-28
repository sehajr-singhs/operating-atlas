"""
3D trajectory visualizations and proper 2D shape plots.

Shows how sensor channels move through time and space — the actual shapes
that the atlas computes on. Each panel is one channel pair traced through time,
coloured by time to show the trajectory's evolution.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
try:
    import figstyle as fs
    HAS_STYLE = True
except ImportError:
    HAS_STYLE = False

# ── colour palette ──────────────────────────────────────────────────
if HAS_STYLE:
    C = fs.C
else:
    C = {'blue': '#0F6493', 'red': '#C4551F', 'green': '#0A7A5E',
         'sky': '#5AABD6', 'grey': '#6C737D', 'orange': '#E8794A'}

FLEET_DIR = os.path.join('C:\\Users\\sehaj\\kaggle_kernel\\out', 'fleet')
OUT = os.path.join(HERE, 'figs_3d_out')
os.makedirs(OUT, exist_ok=True)


def load_unit(data, unit_idx, ep_idx=0):
    """Load one unit's one episode from fleet data."""
    key = f'X_{unit_idx}_{ep_idx}'
    if key in data:
        return data[key]
    return None


def colour_by_time(ax, x, y, cmap='viridis', lw=0.8, alpha=0.7):
    """Plot a trajectory coloured by time using LineCollection."""
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    t = np.linspace(0, 1, len(x))
    lc = LineCollection(segments, cmap=cmap, linewidths=lw, alpha=alpha)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    ax.autoscale()


def fig_3d_trajectories(data, channels, unit_idx=0, ep_idx=0, n_ch=31):
    """3D trajectory: three channels traced through time in 3D space."""
    X = load_unit(data, unit_idx, ep_idx)
    if X is None:
        return
    
    # Pick 3 interesting channels (indices 5=torque, 8=pm_temp, 9=stator_yoke)
    ch_names = [str(c) for c in data['channels']] if 'channels' in data else [f'ch{i}' for i in range(n_ch)]
    
    # Find torque, winding temp, speed channels
    pick = []
    for name in ['torque', 'stator_winding', 'motor_speed']:
        for i, c in enumerate(ch_names):
            if c == name:
                pick.append(i)
                break
    if len(pick) < 3:
        pick = [5, 9, 10]  # fallback
    
    fig = plt.figure(figsize=(14, 6))
    
    # Left: 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    t = np.arange(len(X))
    
    # Subsample for performance
    step = max(1, len(X) // 2000)
    xs, ys, zs = X[::step, pick[0]], X[::step, pick[1]], X[::step, pick[2]]
    ts = t[::step]
    
    # Colour by time
    for i in range(len(xs)-1):
        frac = i / len(xs)
        color = plt.cm.plasma(frac)
        ax1.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2], color=color, lw=0.6, alpha=0.8)
    
    ax1.set_xlabel(ch_names[pick[0]], fontsize=9)
    ax1.set_ylabel(ch_names[pick[1]], fontsize=9)
    ax1.set_zlabel(ch_names[pick[2]], fontsize=9)
    ax1.set_title('3D trajectory through sensor space\n(coloured by time: purple→yellow)', fontsize=10, fontweight='bold')
    ax1.view_init(elev=25, azim=130)
    
    # Right: three 2D projections
    gs = GridSpec(1, 3, figure=fig, left=0.55, right=0.98, wspace=0.35)
    
    pairs = [(0, 1, 'torque vs winding T'), (0, 2, 'torque vs speed'), (1, 2, 'winding T vs speed')]
    for idx, (i, j, title) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, idx])
        xi, xj = X[::step, pick[i]], X[::step, pick[j]]
        colour_by_time(ax, xi, xj, lw=0.7)
        ax.set_xlabel(ch_names[pick[i]], fontsize=8)
        ax.set_ylabel(ch_names[pick[j]], fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Unit {unit_idx}, Episode {ep_idx} — sensor trajectories through time and space',
                 fontsize=12, fontweight='bold', y=0.98)
    
    out = os.path.join(OUT, 'fig_3d_trajectory.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')
    return out


def fig_shape_gallery(data, channels, unit_idx=0, ep_idx=0, n_ch=31):
    """Gallery of channel-pair shapes — each is a different coupling geometry."""
    X = load_unit(data, unit_idx, ep_idx)
    if X is None:
        return
    
    ch_names = [str(c) for c in data['channels']] if 'channels' in data else [f'ch{i}' for i in range(n_ch)]
    
    # Pick interesting pairs
    pairs = [
        (5, 9, 'torque vs winding T\n(oriented hysteresis)'),
        (5, 10, 'torque vs stator yoke T\n(nonlinear coupling)'),
        (9, 10, 'winding T vs yoke T\n(slow co-drift)'),
        (5, 6, 'torque vs speed\n(force-velocity)'),
        (9, 11, 'winding T vs coolant T\n(thermal lag)'),
        (6, 8, 'speed vs magnet T\n(electromagnetic)'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for idx, (i, j, title) in enumerate(pairs):
        if i >= X.shape[1] or j >= X.shape[1]:
            continue
        ax = axes[idx // 3, idx % 3]
        xi, xj = X[:, i], X[:, j]
        
        # Colour by time
        step = max(1, len(xi) // 2000)
        xi_s, xj_s = xi[::step], xj[::step]
        colour_by_time(ax, xi_s, xj_s, cmap='plasma', lw=0.5, alpha=0.6)
        
        ax.set_xlabel(ch_names[i], fontsize=8)
        ax.set_ylabel(ch_names[j], fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)
    
    fig.suptitle(f'Channel-pair shapes traced by Unit {unit_idx} — each is a different physical coupling',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    out = os.path.join(OUT, 'fig_shape_gallery.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')
    return out


def fig_multi_unit_comparison(data, channels, units=[0, 5, 10, 15], ep_idx=0, n_ch=31):
    """Same channel pair across different units — shows what makes each machine unique."""
    ch_names = [str(c) for c in data['channels']] if 'channels' in data else [f'ch{i}' for i in range(n_ch)]
    
    # Find torque and winding temp
    ti, wi = None, None
    for i, c in enumerate(ch_names):
        if c == 'torque': ti = i
        if c == 'stator_winding': wi = i
    if ti is None or wi is None:
        ti, wi = 5, 9
    
    fig, axes = plt.subplots(1, len(units), figsize=(4*len(units), 4))
    if len(units) == 1:
        axes = [axes]
    
    for idx, uid in enumerate(units):
        X = load_unit(data, uid, ep_idx)
        if X is None:
            continue
        ax = axes[idx]
        step = max(1, len(X) // 2000)
        xi, xj = X[::step, ti], X[::step, wi]
        colour_by_time(ax, xi, xj, cmap='plasma', lw=0.5, alpha=0.6)
        ax.set_xlabel('torque', fontsize=9)
        ax.set_ylabel('winding T', fontsize=9)
        ax.set_title(f'Unit {uid}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)
    
    fig.suptitle('Same channel pair, different machines — each traces a unique shape',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out = os.path.join(OUT, 'fig_multi_unit.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')
    return out


def fig_level_vs_shape(data_level, data_shape, channels, unit_idx=0, ep_idx=0, n_ch=31):
    """Show the difference between level fleet and shape fleet trajectories."""
    ch_names = [str(c) for c in channels] if channels is not None else [f'ch{i}' for i in range(n_ch)]
    
    ti, wi = None, None
    for i, c in enumerate(ch_names):
        if c == 'torque': ti = i
        if c == 'stator_winding': wi = i
    if ti is None or wi is None:
        ti, wi = 5, 9
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, data, label in [(axes[0], data_level, 'Level fleet\n(same shape, different magnitude)'),
                             (axes[1], data_shape, 'Shape fleet\n(different coupling geometry)')]:
        if data is None:
            ax.text(0.5, 0.5, 'Data not loaded', ha='center', va='center', transform=ax.transAxes)
            continue
        # Show 5 units
        for uid in range(min(5, 48)):
            X = load_unit(data, uid, ep_idx)
            if X is None:
                continue
            step = max(1, len(X) // 1500)
            xi, xj = X[::step, ti], X[::step, wi]
            # Normalise to show shape, not magnitude
            xi = (xi - xi.mean()) / (xi.std() + 1e-9)
            xj = (xj - xj.mean()) / (xj.std() + 1e-9)
            ax.plot(xi, xj, lw=0.4, alpha=0.5, label=f'Unit {uid}')
        
        ax.set_xlabel('torque (standardised)', fontsize=9)
        ax.set_ylabel('winding T (standardised)', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc='upper right')
    
    fig.suptitle('Why the atlas is blind to level but sensitive to shape',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out = os.path.join(OUT, 'fig_level_vs_shape.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')
    return out


def main():
    print('Loading fleet data...')
    fleet_path = os.path.join(FLEET_DIR, 'ur5e_u80_e6_s90.npz')
    if not os.path.exists(fleet_path):
        # Try other paths
        for p in ['ur5e_u80_e6_s90.npz', 'panda_u80_e6_s90.npz']:
            fp = os.path.join(FLEET_DIR, p)
            if os.path.exists(fp):
                fleet_path = fp
                break
    
    if not os.path.exists(fleet_path):
        print(f'Fleet data not found at {fleet_path}')
        print('Available:', os.listdir(FLEET_DIR) if os.path.exists(FLEET_DIR) else 'dir not found')
        return
    
    data = np.load(fleet_path, allow_pickle=True)
    channels = [str(c) for c in data['channels']]
    print(f'Loaded: {len(channels)} channels, {data["unit_ids"].shape[0]} units')
    
    # Generate all figures
    print('\n--- 3D trajectory ---')
    fig_3d_trajectories(data, channels, unit_idx=0)
    
    print('\n--- Shape gallery ---')
    fig_shape_gallery(data, channels, unit_idx=0)
    
    print('\n--- Multi-unit comparison ---')
    fig_multi_unit_comparison(data, channels, units=[0, 5, 10, 15, 20])
    
    # Load level fleet for comparison
    level_path = os.path.join(FLEET_DIR, 'ur5e_u80_e6_s90.npz')
    shape_path = os.path.join(FLEET_DIR, 'ur5e_u80_e6_s90.npz')
    
    # Check for level/shape specific files
    for p in os.listdir(FLEET_DIR):
        if 'level' in p.lower():
            level_path = os.path.join(FLEET_DIR, p)
        if 'shape' in p.lower():
            shape_path = os.path.join(FLEET_DIR, p)
    
    data_level = np.load(level_path, allow_pickle=True) if os.path.exists(level_path) else None
    data_shape = np.load(shape_path, allow_pickle=True) if os.path.exists(shape_path) else None
    
    print('\n--- Level vs Shape ---')
    fig_level_vs_shape(data_level, data_shape, channels)
    
    print(f'\nAll figures saved to {OUT}')


if __name__ == '__main__':
    main()
