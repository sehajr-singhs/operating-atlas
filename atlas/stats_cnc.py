"""
Bootstrap confidence intervals and significance tests for the CNC results.

We compute LOO predictions once (expensive), then bootstrap the R^2 and
accuracy statistics from those predictions (cheap). The permutation test
swaps geometry/marginal predictions per sample.

This gives us:
- 95% CI on R^2 for feedrate (geometry vs marginals)
- Paired permutation p-value (geometry vs marginals)
- Wilcoxon signed-rank p-value as a check
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, wilcoxon
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import ioo_chart as ic
import manifold_local as ml

P = os.path.expanduser('~/probe/cnc')
B = 10000  # bootstrap iterations (fast: just resampling residuals)


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


def loo_predict(F, y, clf=False):
    F = np.nan_to_num(F)
    nc = int(min(8, F.shape[0] - 2, F.shape[1]))
    if clf:
        pipe = make_pipeline(StandardScaler(), PCA(n_components=nc),
                             LogisticRegression(max_iter=5000))
        return cross_val_predict(pipe, F, y, cv=LeaveOneOut(), method='predict')
    else:
        pipe = make_pipeline(StandardScaler(), PCA(n_components=nc),
                             RidgeCV(alphas=np.logspace(-2, 5, 30)))
        return cross_val_predict(pipe, F, y, cv=LeaveOneOut())


def r2_from_pred(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def bootstrap_r2(y, p, B=10000, seed=42):
    """Bootstrap CI on R^2 by resampling the (y, pred) pairs."""
    rng = np.random.default_rng(seed)
    n = len(y)
    r2s = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        r2s.append(r2_from_pred(y[idx], p[idx]))
    return np.percentile(r2s, [2.5, 50, 97.5])


def paired_perm_test(pred_g, pred_m, y, B=10000, seed=42):
    """Paired permutation test on absolute error."""
    rng = np.random.default_rng(seed)
    err_g = np.abs(y - pred_g)
    err_m = np.abs(y - pred_m)
    obs = np.mean(err_m - err_g)
    count = 0
    for _ in range(B):
        mask = rng.integers(0, 2, len(y))
        e1 = np.where(mask == 1, err_g, err_m)
        e2 = np.where(mask == 1, err_m, err_g)
        if np.mean(e2 - e1) >= obs:
            count += 1
    return count / B


def main():
    runs, meta, cols = load()
    n = len(runs)
    print(f'CNC mill: {n} runs, {runs[0].shape[1]} channels')
    print(f'Bootstrap B={B}\n')

    chart = ic.OperatorChart(n_landmarks=90, k=40).fit_class(runs)
    ioos = [chart.ioo(X) for X in runs]
    chart.set_core(ioos, min_frac=0.85)
    G = np.stack([chart.descriptor(i) for i in ioos])
    M = np.stack([marginal(X) for X in runs])

    # ---- Feedrate ----
    y = meta['feedrate'].to_numpy(float)
    pred_g = loo_predict(G, y)
    pred_m = loo_predict(M, y)
    r2_g = r2_from_pred(y, pred_g)
    r2_m = r2_from_pred(y, pred_m)
    ci_g = bootstrap_r2(y, pred_g, B)
    ci_m = bootstrap_r2(y, pred_m, B)
    p_perm = paired_perm_test(pred_g, pred_m, y, B)
    try:
        _, p_w = wilcoxon(np.abs(y - pred_g), np.abs(y - pred_m))
    except Exception:
        p_w = float('nan')
    rho_g = np.corrcoef(y, pred_g)[0, 1]
    rho_m = np.corrcoef(y, pred_m)[0, 1]

    print('=== Feedrate (continuous target) ===')
    print(f'  Geometry:  R2 = {r2_g:.3f}  [{ci_g[0]:.3f}, {ci_g[2]:.3f}]  '
          f'r = {rho_g:.3f}')
    print(f'  Marginals: R2 = {r2_m:.3f}  [{ci_m[0]:.3f}, {ci_m[2]:.3f}]  '
          f'r = {rho_m:.3f}')
    print(f'  Permutation p = {p_perm:.4f}  (one-sided)')
    print(f'  Wilcoxon p    = {p_w:.4f}')
    print()

    # ---- Clamp pressure ----
    y2 = meta['clamp_pressure'].to_numpy(float)
    pred_g2 = loo_predict(G, y2)
    pred_m2 = loo_predict(M, y2)
    r2_g2 = r2_from_pred(y2, pred_g2)
    r2_m2 = r2_from_pred(y2, pred_m2)
    ci_g2 = bootstrap_r2(y2, pred_g2, B)
    ci_m2 = bootstrap_r2(y2, pred_m2, B)

    print('=== Clamp pressure (continuous target) ===')
    print(f'  Geometry:  R2 = {r2_g2:.3f}  [{ci_g2[0]:.3f}, {ci_g2[2]:.3f}]')
    print(f'  Marginals: R2 = {r2_m2:.3f}  [{ci_m2[0]:.3f}, {ci_m2[2]:.3f}]')
    print()

    # ---- Tool wear ----
    y3 = (meta['tool_condition'] == 'worn').astype(int)
    pred_g3 = loo_predict(G, y3, clf=True)
    pred_m3 = loo_predict(M, y3, clf=True)
    acc_g = (pred_g3 == y3).mean()
    acc_m = (pred_m3 == y3).mean()
    maj = max(np.mean(y3 == c) for c in np.unique(y3))

    print('=== Tool wear (binary target) ===')
    print(f'  Geometry:  acc = {acc_g*100:.1f}%')
    print(f'  Marginals: acc = {acc_m*100:.1f}%')
    print(f'  Majority:  {maj*100:.1f}%')
    print()
    print('Neither method clears majority class => no claim.')

    # ---- Summary for the paper ----
    print('\n=== SUMMARY FOR PAPER ===')
    print(f'Feedrate: geometry R2={r2_g:.2f} [{ci_g[0]:.2f}, {ci_g[2]:.2f}], '
          f'marginals R2={r2_m:.2f} [{ci_m[0]:.2f}, {ci_m[2]:.2f}], '
          f'p={p_perm:.4f} (perm), p={p_w:.4f} (Wilcoxon)')
    print(f'Clamp:    geometry R2={r2_g2:.2f} [{ci_g2[0]:.2f}, {ci_g2[2]:.2f}], '
          f'marginals R2={r2_m2:.2f} [{ci_m2[0]:.2f}, {ci_m2[2]:.2f}]')
    print(f'Tool:     geometry acc={acc_g*100:.1f}%, '
          f'marginals acc={acc_m*100:.1f}%, majority={maj*100:.1f}%')


if __name__ == '__main__':
    main()
