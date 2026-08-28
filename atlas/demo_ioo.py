"""
The IOO loop end to end on the real motor bench, and a picture of the body.

Sessions of the bench act as the "devices" here. That is a scope limit stated
plainly: the public bench is one machine, so what is fitted is a base body for
the class and per-session deformations of it. The machinery is identical when
the units are genuinely distinct machines; only the data would differ.

Three things are checked, in order of how much they would hurt if false:

  1  the base body plus a low-dimensional deformation reconstructs a held-out
     device better than the base body alone. If not, the "points shift to fit
     your device" step is doing nothing.
  2  the deformation code is stable, so the same device fitted from two
     disjoint halves of its record lands in the same place.
  3  off-body residual rises when the machine leaves the envelope its class
     occupies, which is what makes the object usable as a prior.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import figstyle as fs        # noqa: E402
import manifold_local as ml  # noqa: E402
import ioo as IOO            # noqa: E402

CACHE = os.path.join(HERE, '_pmsm_sessions.npz')


def main():
    fs.setup()
    z = np.load(CACHE)
    X_all, pids = z['X'], z['pids']
    CH = [str(c) for c in z['channels']]
    units = [X_all[i].astype(np.float64) for i in range(len(X_all))]
    print(f'{len(units)} sessions, {units[0].shape[1]} channels')

    n_hold = 10
    train, test = units[:-n_hold], units[-n_hold:]

    idx = IOO.OperatorIndex()
    info = idx.add_class('motor', train, n_landmarks=140)
    print('base IOO for class "motor":', info)

    # ---- 1 does the deformation earn its place? -------------------------
    # The code is fitted on the FIRST half of a device's record and scored on
    # the SECOND. Fitting and scoring on the same samples would make this claim
    # true by construction: the fitted landmarks are built from those very
    # points, so of course they sit closer to them.
    base_err, fit_err = [], []
    for X in test:
        h = len(X) // 2
        f = idx.fit_unit('motor', X[:h])
        Z_eval = IOO._embed(X[h:])[0][:, :f['base'].shape[1]]
        _, r_base = idx.project({'landmarks': f['base']}, Z_eval)
        _, r_fit = idx.project(f, Z_eval)
        base_err.append(np.median(r_base)); fit_err.append(np.median(r_fit))
    base_err, fit_err = np.array(base_err), np.array(fit_err)
    gain = 100 * (1 - fit_err.mean() / base_err.mean())
    print(f'\n1  held-out distance to the body: base {base_err.mean():.4f} '
          f'-> fitted {fit_err.mean():.4f}   ({gain:+.1f}%)')
    print(f'   improved on {int((fit_err < base_err).sum())}/{len(test)} devices')

    # ---- 2 is the code a property of the device? ------------------------
    same, diff = [], []
    codes_a, codes_b = [], []
    for X in test:
        h = len(X) // 2
        a = idx.fit_unit('motor', X[:h])['code_norm']
        b = idx.fit_unit('motor', X[h:])['code_norm']
        codes_a.append(a); codes_b.append(b)
    codes_a, codes_b = np.array(codes_a), np.array(codes_b)
    for i in range(len(test)):
        same.append(np.linalg.norm(codes_a[i] - codes_b[i]))
        for j in range(len(test)):
            if i != j:
                diff.append(np.linalg.norm(codes_a[i] - codes_b[j]))
    same, diff = np.array(same), np.array(diff)
    S = codes_a @ codes_b.T
    rank = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1)
    print(f'\n2  code distance, same device {same.mean():.3f} vs different '
          f'{diff.mean():.3f}   (ratio {diff.mean()/max(same.mean(),1e-9):.2f}x)')
    print(f'   half-to-half retrieval top-1 = {100*(rank==0).mean():.0f}% '
          f'of {len(test)}, chance {100/len(test):.0f}%')

    # ---- 3 does leaving the envelope show up? ---------------------------
    f0 = idx.fit_unit('motor', test[0])
    _, r_in = idx.project(f0, f0['Z'])
    rng = np.random.default_rng(0)
    Zout = f0['Z'] + rng.normal(0, 0.25, f0['Z'].shape)
    _, r_out = idx.project(f0, Zout)
    print(f'\n3  off-body residual: on-envelope {np.median(r_in):.4f}, '
          f'pushed off {np.median(r_out):.4f}   '
          f'({np.median(r_out)/max(np.median(r_in),1e-9):.1f}x)')

    # ---- the picture -----------------------------------------------------
    geo = ml.local_geometry(test[0], k=48, n_probe=2500, seed=0)
    Zp = geo['Z'][geo['probe']]
    _, _, Vt = np.linalg.svd(Zp - Zp.mean(0), full_matrices=False)
    E = (Zp - Zp.mean(0)) @ Vt[:3].T

    fig = plt.figure(figsize=(12.4, 7.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.26,
                  left=0.04, right=0.97, top=0.88, bottom=0.07)
    fields = [('dim', 'local dimension', 'viridis'),
              ('curv', 'curvature', 'magma'),
              ('tear', 'tear / boundary', 'cividis')]
    for c, (key, lab, cmap) in enumerate(fields):
        ax = fig.add_subplot(gs[0, c], projection='3d')
        v = geo['fields'][key]
        lo, hi = np.nanpercentile(v, [3, 97])
        p = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=np.clip(v, lo, hi),
                       cmap=cmap, s=2.2, lw=0, alpha=.75)
        ax.set_title(f'{"abc"[c]}   {lab}', loc='left', fontsize=9.5)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        fig.colorbar(p, ax=ax, shrink=.55, pad=.02).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(geo['fields']['dim'], bins=44, color=fs.C['blue'])
    ax.set_xlabel('local dimension'); ax.set_ylabel('operating points')
    ax.set_title('d   the body is not one dimension', loc='left', fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    B, F = f0['base'], f0['landmarks']
    Eb = (B - Zp.mean(0)) @ Vt[:2].T
    Ef = (F - Zp.mean(0)) @ Vt[:2].T
    ax.scatter(E[:, 0], E[:, 1], s=1.4, color=fs.C['grey'], alpha=.22, lw=0)
    ax.quiver(Eb[:, 0], Eb[:, 1], Ef[:, 0] - Eb[:, 0], Ef[:, 1] - Eb[:, 1],
              angles='xy', scale_units='xy', scale=1, width=.0032,
              color=fs.C['red'], alpha=.85)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('e   the base body, and how its landmarks\n     shift to fit '
                 'this device', loc='left', fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(np.arange(len(f0['code_norm'])), f0['code_norm'], color=fs.C['red'])
    ax.axhline(0, color=fs.C['black'], lw=.7)
    ax.set_xlabel('deformation direction'); ax.set_ylabel('code (sd)')
    ax.set_title('f   the fitted code is the unit identity', loc='left',
                 fontsize=9.5)

    fig.suptitle('The Index of Operations: one device\'s operating body, and '
                 'the deformation that fits the class base to it', y=.955,
                 fontsize=11)
    out = os.path.join(HERE, 'fig5_ioo.png')
    fig.savefig(out, dpi=190)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
