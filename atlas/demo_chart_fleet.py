"""
Local geometry on a shared chart, tested against the same three claims that the
landmark-displacement parameterisation failed.

Baselines are carried so a win cannot be read as a win over nothing:
  displacement  the previous parameterisation, which failed
  raw marginals per-channel mean/sd/skew/kurtosis, the strong simple baseline
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ioo_chart as ic                              # noqa: E402
from scipy.stats import skew, kurtosis              # noqa: E402
from sklearn.linear_model import RidgeCV            # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline          # noqa: E402
from sklearn.preprocessing import StandardScaler    # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402

FLEET = os.environ.get(
    'FLEET', os.path.expanduser('~/kaggle_kernel/out/fleet/ur5e_u80_e6_s90.npz'))
IDENT = ['payload', 'r_wh', 'r_ha', 'k_cu', 'damp', 'gain', 'skew']
MAXN = int(os.environ.get('MAXN', 8000))


def load(path):
    z = np.load(path, allow_pickle=False)
    eps = {}
    for k in z.files:
        if k.startswith('X_'):
            _, u, e = k.split('_')
            eps.setdefault(int(u), {})[int(e)] = z[k]
    return z['idents'], eps


def stack(eu, keys, maxn=MAXN):
    X = np.concatenate([eu[k] for k in keys]).astype(np.float64)
    return X[::max(1, len(X) // maxn)][:maxn]


def marginal(X):
    return np.nan_to_num(np.concatenate(
        [X.mean(0), X.std(0), skew(X, 0), kurtosis(X, 0)]))


def decode(F, Y, tag):
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=int(min(24, F.shape[0] // 4,
                                                  F.shape[1]))),
                         RidgeCV(alphas=np.logspace(-2, 5, 30)))
    cv = KFold(5, shuffle=True, random_state=0)
    out = {}
    for j, nm in enumerate(IDENT):
        y = Y[:, j]
        if y.std() < 1e-9:
            out[nm] = float('nan'); continue
        p = cross_val_predict(pipe, F, y, cv=cv)
        out[nm] = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f'  {tag:<26} ' + '  '.join(f'{k}={v:+.2f}' for k, v in out.items()))
    return out


def retrieval(A, B, tag):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = An @ Bn.T
    r = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1)
    print(f'  {tag:<26} top-1 {100*(r==0).mean():5.1f} %   '
          f'pct-rank {100*(1-r/(len(S)-1)).mean():5.1f} %   '
          f'(chance {100/len(S):.1f} %)')
    return float((r == 0).mean())


def main():
    idents, eps = load(FLEET)
    plat = os.path.basename(FLEET).split('_')[0]
    units = sorted(eps)
    print(f'{plat}: {len(units)} distinct machines, '
          f'{len(eps[units[0]])} episodes each')

    rng = np.random.default_rng(0)
    order = list(rng.permutation(units))
    train_u, test_u = order[20:], order[:20]

    halves = {}
    for u in units:
        ks = sorted(eps[u])
        h = len(ks) // 2
        halves[u] = (stack(eps[u], ks[:h]), stack(eps[u], ks[h:]))

    chart = ic.OperatorChart(n_landmarks=120, k=48).fit_class(
        [halves[u][0] for u in train_u])
    print(f'chart: {chart.K} landmarks, {chart.chart_.shape[1]} live channels, '
          f'coverage radius {chart.cover_r_:.4f}')

    ioos_a = {u: chart.ioo(halves[u][0]) for u in units}
    chart.set_core([ioos_a[u] for u in train_u], min_frac=0.9)
    ioos_b = {u: chart.ioo(halves[u][1]) for u in units}

    cov = np.mean([i['covered'].mean() for i in ioos_a.values()])
    print(f'mean chart coverage per device: {100*cov:.0f} %')

    A = np.stack([chart.descriptor(ioos_a[u]) for u in test_u])
    B = np.stack([chart.descriptor(ioos_b[u]) for u in test_u])
    Am = np.stack([marginal(halves[u][0]) for u in test_u])
    Bm = np.stack([marginal(halves[u][1]) for u in test_u])

    print('\n2  is the descriptor the device? (fit each half separately)')
    retrieval(A, B, 'geometry on the chart')
    retrieval(Am, Bm, 'per-channel marginals')

    print('\n3  decoding the TRUE physical parameters (all 80 machines)')
    Y = np.stack([idents[u] for u in units])
    Fg = np.stack([chart.descriptor(ioos_a[u]) for u in units])
    Fm = np.stack([marginal(halves[u][0]) for u in units])
    decode(Fg, Y, 'geometry on the chart')
    decode(Fm, Y, 'per-channel marginals')


if __name__ == '__main__':
    main()
