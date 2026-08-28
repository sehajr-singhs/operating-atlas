"""
The IOO on a machine that actually has a body.

Everything before this ran on telemetry that fills most of its channel count,
where there is no low-dimensional shape to find. A CNC mill running the same
part program under closed-loop servo control is the opposite case: 35 live
channels collapse to about 7 dimensions, because the servos force actual axis
state to track commanded state and the program forces the path.

The dataset is 18 runs of one program, each labelled with

    feedrate         3 to 20        a commanded operating condition
    clamp_pressure   2.5 to 4       a fixture condition
    tool_condition   worn / unworn  the fault

So the class base is the part program's body, and each run is that body deformed
by its own feedrate, clamping and tool state. The question is whether the
deformation carries those three, and whether it carries them better than reading
the channels one at a time.

n = 18, so everything is leave-one-out and no result here is worth more than the
sample size allows. It is reported with that in front rather than behind.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import ioo_chart as ic                                   # noqa: E402
import manifold_local as ml                              # noqa: E402
from scipy.stats import skew, kurtosis                   # noqa: E402
from sklearn.linear_model import RidgeCV, LogisticRegression  # noqa: E402
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline               # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402
from sklearn.decomposition import PCA                    # noqa: E402

P = os.path.expanduser('~/probe/cnc')


def load():
    meta = pd.read_csv(os.path.join(P, 'train.csv'))
    runs, keep = [], []
    d0 = pd.read_csv(os.path.join(P, 'experiment_01.csv'))
    cols = [c for c in d0.columns if c != 'Machining_Process'
            and pd.to_numeric(d0[c], errors='coerce').notna().mean() > 0.9]
    for i in range(1, 19):
        f = os.path.join(P, f'experiment_{i:02d}.csv')
        if not os.path.exists(f):
            continue
        X = pd.read_csv(f)[cols].to_numpy(float)
        X = X[np.isfinite(X).all(1)]
        if len(X) < 400:
            continue
        runs.append(X); keep.append(i - 1)
    return runs, meta.iloc[keep].reset_index(drop=True), cols


def marginal(X):
    return np.nan_to_num(np.concatenate(
        [X.mean(0), X.std(0), skew(X, 0), kurtosis(X, 0)]))


def reg(F, y, name, tag):
    F = np.nan_to_num(F)
    nc = int(min(8, F.shape[0] - 2, F.shape[1]))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=nc),
                         RidgeCV(alphas=np.logspace(-2, 5, 30)))
    p = cross_val_predict(pipe, F, y, cv=LeaveOneOut())
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rho = np.corrcoef(y, p)[0, 1] if np.std(p) > 1e-9 else np.nan
    print(f'    {name:<10} {tag:<22} R2={r2:+.2f}  r={rho:+.2f}')
    return r2


def clf(F, y, name, tag):
    F = np.nan_to_num(F)
    nc = int(min(8, F.shape[0] - 2, F.shape[1]))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=nc),
                         LogisticRegression(max_iter=5000))
    s = cross_val_score(pipe, F, y, cv=LeaveOneOut())
    maj = max(np.mean(y == c) for c in np.unique(y))
    print(f'    {name:<10} {tag:<22} acc={s.mean()*100:5.1f} %  '
          f'(majority {maj*100:.0f} %)')
    return s.mean()


def main():
    runs, meta, cols = load()
    print(f'{len(runs)} runs of one part program, {runs[0].shape[1]} channels, '
          f'{runs[0].shape[0]}-{max(len(r) for r in runs)} samples each')

    Zall = np.concatenate(runs)
    zz = ml._ranks(Zall)
    print(f'intrinsic dimension of the pooled body: '
          f'TwoNN = {ml.intrinsic_dim_twonn(zz[::3]):.2f} of {zz.shape[1]}')

    chart = ic.OperatorChart(n_landmarks=90, k=40).fit_class(runs)
    ioos = [chart.ioo(X) for X in runs]
    chart.set_core(ioos, min_frac=0.85)
    print(f'chart: {chart.K} landmarks, core {chart.core_.sum()} '
          f'({100*chart.core_frac_:.0f} % of the chart visited by >=85 % of runs)')

    G = np.stack([chart.descriptor(i) for i in ioos])
    M = np.stack([marginal(X) for X in runs])
    print(f'descriptors: geometry {G.shape}, marginals {M.shape}')

    print('\ndecoding the run conditions (leave-one-out, n=%d)' % len(runs))
    for col, kind in [('feedrate', 'reg'), ('clamp_pressure', 'reg'),
                      ('tool_condition', 'clf')]:
        y = meta[col].to_numpy()
        print(f'  -- {col} --')
        if kind == 'reg':
            y = y.astype(float)
            reg(G, y, 'geometry', 'on the class chart')
            reg(M, y, 'marginals', 'per channel')
        else:
            y = (y == 'worn').astype(int)
            clf(G, y, 'geometry', 'on the class chart')
            clf(M, y, 'marginals', 'per channel')


if __name__ == '__main__':
    main()
