"""
The IOO deformation on a fleet of genuinely distinct machines.

The motor bench could not test this. It is one machine, and its sessions each
sweep a different sub-region of the envelope, so there is no common body for
each "device" to deform: a landmark one session never visits has no meaningful
position for that session. The fleets do have what is needed, 80 distinct
machines per platform, each run over several episodes, and each with a known
physical identity that was never shown to the method.

Three claims, in order of how much they cost if false:

  1  DEFORMATION EARNS ITS PLACE. Fit the code on a held-out device's early
     episodes, score the distance to the body on its LATER episodes. If the
     fitted body is no closer than the undeformed class base, the shift step is
     doing nothing.
  2  THE CODE IS THE DEVICE. Fit the same machine twice from disjoint episodes
     and it should land in the same place, and closer to itself than to its
     siblings.
  3  THE CODE IS PHYSICAL. Regress the true payload, thermal resistances,
     winding resistance and bearing friction on the code. This is the test no
     public fleet permits, and it is the one that says whether the shift is
     tracking the machine or the workload.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ioo as IOO                                   # noqa: E402
from sklearn.linear_model import RidgeCV            # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline          # noqa: E402
from sklearn.preprocessing import StandardScaler    # noqa: E402

FLEET = os.environ.get(
    'FLEET', os.path.expanduser('~/kaggle_kernel/out/fleet/ur5e_u80_e6_s90.npz'))
IDENT = ['payload', 'r_wh', 'r_ha', 'k_cu', 'damp', 'gain', 'skew']
MAXN = int(os.environ.get('MAXN', 9000))


def load(path):
    z = np.load(path, allow_pickle=False)
    idents = z['idents']
    eps = {}
    for k in z.files:
        if k.startswith('X_'):
            _, u, e = k.split('_')
            eps.setdefault(int(u), {})[int(e)] = z[k]
    return idents, eps


def stack(eps_u, keys, maxn=MAXN):
    X = np.concatenate([eps_u[k] for k in keys]).astype(np.float64)
    return X[::max(1, len(X) // maxn)][:maxn]


def main():
    idents, eps = load(FLEET)
    plat = os.path.basename(FLEET).split('_')[0]
    units = sorted(eps)
    print(f'{plat}: {len(units)} distinct machines, '
          f'{len(eps[units[0]])} episodes each')

    rng = np.random.default_rng(0)
    order = rng.permutation(units)
    n_test = 20
    test_u, train_u = list(order[:n_test]), list(order[n_test:])

    # class base from the training machines, early episodes only
    tr = []
    for u in train_u:
        ks = sorted(eps[u])
        tr.append(stack(eps[u], ks[:len(ks) // 2]))
    idx = IOO.OperatorIndex()
    info = idx.add_class(plat, tr, n_landmarks=150)
    print('base IOO:', info)

    # ---- 1 does the deformation earn its place? -------------------------
    base_err, fit_err, codes_a, codes_b = [], [], [], []
    for u in test_u:
        ks = sorted(eps[u])
        h = len(ks) // 2
        Xa, Xb = stack(eps[u], ks[:h]), stack(eps[u], ks[h:])
        f = idx.fit_unit(plat, Xa)
        d = f['base'].shape[1]
        Zb = IOO._embed(Xb)[0]
        Zb = Zb[:, :d] if Zb.shape[1] >= d else np.pad(
            Zb, ((0, 0), (0, d - Zb.shape[1])))
        _, rb = idx.project({'landmarks': f['base']}, Zb)
        _, rf = idx.project(f, Zb)
        base_err.append(np.median(rb)); fit_err.append(np.median(rf))
        codes_a.append(f['code_norm'])
        codes_b.append(idx.fit_unit(plat, Xb)['code_norm'])
    base_err, fit_err = np.array(base_err), np.array(fit_err)
    print(f'\n1  held-out episodes, distance to the body: '
          f'base {base_err.mean():.4f} -> fitted {fit_err.mean():.4f}   '
          f'({100*(1-fit_err.mean()/base_err.mean()):+.1f}%)')
    print(f'   fitted body closer on {int((fit_err < base_err).sum())}/{n_test} '
          f'machines')

    # ---- 2 is the code the device? --------------------------------------
    A, B = np.array(codes_a), np.array(codes_b)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = An @ Bn.T
    rank = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1)
    same = np.linalg.norm(A - B, axis=1).mean()
    diff = np.mean([np.linalg.norm(A[i] - B[j])
                    for i in range(len(A)) for j in range(len(A)) if i != j])
    print(f'\n2  same machine {same:.3f} vs different {diff:.3f} '
          f'({diff/max(same,1e-9):.2f}x);  retrieval top-1 '
          f'{100*(rank==0).mean():.0f}%  (chance {100/n_test:.0f}%)')

    # ---- 3 is the code physical? ----------------------------------------
    print('\n3  decoding the TRUE physical parameters from the code')
    allc, Y = [], []
    for u in units:
        ks = sorted(eps[u])
        allc.append(idx.fit_unit(plat, stack(eps[u], ks[:len(ks) // 2]))['code'])
        Y.append(idents[u])
    C, Y = np.array(allc), np.array(Y)
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30)))
    cv = KFold(5, shuffle=True, random_state=0)
    for j, nm in enumerate(IDENT):
        y = Y[:, j]
        if y.std() < 1e-9:
            continue
        p = cross_val_predict(pipe, C, y, cv=cv)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f'   {nm:<9} R2 = {r2:+.2f}')


if __name__ == '__main__':
    main()
