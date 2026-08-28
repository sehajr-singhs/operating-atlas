"""
Real baseline comparisons on the CNC mill, so the paper's Discussion has
numbers instead of arguments.

Decode (feedrate from a per-run descriptor, leave-one-out, n=18):
    chart geometry (the paper's method)
    per-channel marginals (the paper's baseline)
    Isomap coordinates of the class body
    diffusion-map coordinates of the class body
    autoencoder bottleneck activations

Denoising (restore corrupted held-out telemetry, metric = mean Euclidean
distance to the true state in the embedded space):
    body kNN (the paper's method)
    PCA rank 8 (the paper's linear baseline)
    median filter (per-channel)
    diffusion-map kNN in diffusion coordinates (Nystrom extension)
    autoencoder reconstruction (bottleneck 8)
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.ndimage import median_filter
from scipy import sparse
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import ioo_chart as ic
import manifold_local as ml

P = os.path.expanduser('~/probe/cnc')
RANK = 8
torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# diffusion map with Nystrom out-of-sample extension
# ---------------------------------------------------------------------------
class DiffusionMap:
    def __init__(self, n_comp=8, k=15):
        self.n_comp = n_comp
        self.k = k

    def fit(self, Z):
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Z)
        dist, idx = nn.kneighbors(Z)
        idx, dist = idx[:, 1:], dist[:, 1:]
        n = len(Z)
        W = np.zeros((n, n))
        self.eps = np.median(dist) ** 2 + 1e-12
        for i in range(n):
            W[i, idx[i]] = np.exp(-dist[i] ** 2 / self.eps)
        W = (W + W.T) / 2
        self.D = W.sum(1) + 1e-12
        Dinv = 1.0 / np.sqrt(self.D)
        S = Dinv[:, None] * W * Dinv[None, :]
        evals, evecs = np.linalg.eigh(S)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        keep = slice(1, self.n_comp + 1)
        self.evals = evals[keep]
        self.evecs = evecs[:, keep]
        self.Z = Z
        self.coords = self._coords(evecs[:, keep], self.evals)
        return self

    def _coords(self, V, ev):
        c = V * (ev ** 1.0)[None, :]
        c = c / self.D[:, None]
        return c

    def transform(self, Y):
        """Nystrom out-of-sample extension, vectorised with sparse rows."""
        nn = NearestNeighbors(n_neighbors=self.k).fit(self.Z)
        dist, idx = nn.kneighbors(Y)
        rows, cidx, vals = [], [], []
        for i in range(len(Y)):
            rows.extend([i] * self.k)
            cidx.extend(idx[i])
            vals.extend(np.exp(-dist[i] ** 2 / self.eps))
        Ky = sparse.csr_matrix((vals, (rows, cidx)),
                               shape=(len(Y), len(self.Z)))
        Dinv = 1.0 / np.sqrt(self.D)
        psi = Ky.multiply(Dinv[None, :])
        coords = psi @ self.evecs / self.evals[None, :]
        coords = np.asarray(coords)
        coords /= self.D[:len(Y), None]
        return coords


def run_descriptors(runs, embedder):
    """Describe each run by quantiles of its embedded points."""
    out = []
    for X in runs:
        Z = embedder(X)
        out.append(np.concatenate([np.quantile(Z[:, j], np.linspace(.05, .95, 9))
                                   for j in range(Z.shape[1])]))
    return np.stack(out)


def autoencoder_fit(X, n_latent=8, epochs=120, bs=512):
    """Train a bottleneck AE on clean X (subsampled for speed). Return model."""
    rng = np.random.default_rng(0)
    if len(X) > 6000:
        X = X[rng.choice(len(X), 6000, replace=False)]
    Xt = torch.tensor(X, dtype=torch.float32)
    d = X.shape[1]
    model = nn.Sequential(
        nn.Linear(d, 64), nn.ReLU(),
        nn.Linear(64, n_latent),
        nn.Linear(n_latent, 64), nn.ReLU(),
        nn.Linear(64, d))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = Xt[perm[i:i + bs]]
            opt.zero_grad()
            loss = lossf(model(b), b)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
    return model


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------
def loo_r2(F, y):
    F = np.nan_to_num(F)
    nc = int(min(8, F.shape[0] - 2, F.shape[1]))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=nc),
                         RidgeCV(alphas=np.logspace(-2, 5, 30)))
    p = cross_val_predict(pipe, F, y, cv=LeaveOneOut())
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def decode():
    runs, meta, cols = load()
    y = meta['feedrate'].to_numpy(float)
    print(f'decode: n={len(runs)} runs, feedrate, leave-one-out')

    pool = np.concatenate(runs)
    rng = np.random.default_rng(1)
    fit_idx = np.sort(rng.choice(len(pool), 4000, replace=False))
    pool_fit = pool[fit_idx]
    grid = np.linspace(0, 1, 512)
    knots = [np.quantile(pool[:, j], grid) for j in range(pool.shape[1])]
    w = np.sqrt(np.maximum(ml.channel_snr(pool) - 0.02, 0.0))
    live = w > 1e-6
    wl = w[live]

    def emb(X):
        Z = np.stack([np.interp(X[:, j], knots[j], grid)
                      for j in range(X.shape[1])], 1)
        return Z[:, live] * wl

    chart = ic.OperatorChart(n_landmarks=90, k=40).fit_class(runs)
    ioos = [chart.ioo(X) for X in runs]
    chart.set_core(ioos, min_frac=0.85)
    G = np.stack([chart.descriptor(i) for i in ioos])
    M = np.stack([marginal(X) for X in runs])

    Ze_fit = emb(pool_fit)
    iso = Isomap(n_components=8, n_neighbors=15).fit(Ze_fit)
    Fiso = run_descriptors(runs, lambda X: iso.transform(emb(X)))

    dm = DiffusionMap(n_comp=8, k=15).fit(Ze_fit)
    Fdm = run_descriptors(runs, lambda X: dm.transform(emb(X)))

    ae = autoencoder_fit(Ze, n_latent=8)
    enc = ae[:3]

    def aeb(X):
        with torch.no_grad():
            return enc(torch.tensor(emb(X), dtype=torch.float32)).numpy()

    Fae = run_descriptors(runs, aeb)

    results = []
    for name, F in [('chart geometry', G), ('marginals', M),
                    ('isomap', Fiso), ('diffusion map', Fdm),
                    ('autoencoder', Fae)]:
        r2 = loo_r2(F, y)
        results.append((name, r2))
        print(f'    {name:<16} R2 = {r2:+.3f}')

    def perm_p(pred_g, pred_b, yy, B=10000):
        rng = np.random.default_rng(42)
        eg, eb = np.abs(yy - pred_g), np.abs(yy - pred_b)
        obs = np.mean(eb - eg)
        cnt = 0
        for _ in range(B):
            m = rng.integers(0, 2, len(yy))
            s1 = np.where(m == 1, eg, eb); s2 = np.where(m == 1, eb, eg)
            if np.mean(s2 - s1) >= obs:
                cnt += 1
        return cnt / B

    F = np.nan_to_num(G)
    nc = int(min(8, F.shape[0] - 2, F.shape[1]))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=nc),
                         RidgeCV(alphas=np.logspace(-2, 5, 30)))
    pred_g = cross_val_predict(pipe, F, y, cv=LeaveOneOut())
    print('\n    geometry vs baseline (paired permutation, B=10000):')
    for name, Fb in [('marginals', M), ('isomap', Fiso),
                     ('diffusion map', Fdm), ('autoencoder', Fae)]:
        Fb = np.nan_to_num(Fb)
        nb = int(min(8, Fb.shape[0] - 2, Fb.shape[1]))
        pipeb = make_pipeline(StandardScaler(), PCA(n_components=nb),
                              RidgeCV(alphas=np.logspace(-2, 5, 30)))
        pred_b = cross_val_predict(pipeb, Fb, y, cv=LeaveOneOut())
        p = perm_p(pred_g, pred_b, y)
        print(f'    vs {name:<16} p = {p:.4f}')
    return results


# ---------------------------------------------------------------------------
# denoising
# ---------------------------------------------------------------------------
def restore_knn(train_Z, Y, k=12):
    nn = NearestNeighbors(n_neighbors=k).fit(train_Z)
    dist, idx = nn.kneighbors(Y)
    wgt = 1.0 / (dist + 1e-9)
    wgt /= wgt.sum(1, keepdims=True)
    return np.einsum('nk,nkd->nd', wgt, train_Z[idx])


def denoise():
    runs, cols = load()
    train, test = runs[:12], runs[12:]
    pool = np.concatenate(train)
    grid = np.linspace(0, 1, 512)
    knots = [np.quantile(pool[:, j], grid) for j in range(pool.shape[1])]
    w = np.sqrt(np.maximum(ml.channel_snr(pool) - 0.02, 0.0))
    live = w > 1e-6
    wl = w[live]

    def emb(X):
        Z = np.stack([np.interp(X[:, j], knots[j], grid)
                      for j in range(X.shape[1])], 1)
        return Z[:, live] * wl

    Ztr = emb(pool)
    rng = np.random.default_rng(1)
    fit_idx = np.sort(rng.choice(len(Ztr), 4000, replace=False))
    Ztr_fit = Ztr[fit_idx]
    pca = PCA(n_components=RANK).fit(Ztr_fit)
    dm = DiffusionMap(n_comp=8, k=15).fit(Ztr_fit)
    ae = autoencoder_fit(Ztr_fit, n_latent=8)
    ae.eval()
    Ztr_t = torch.tensor(Ztr, dtype=torch.float32)

    print(f'\ndenoi: {len(train)} train / {len(test)} test runs, '
          f'{Ztr.shape[1]} live channels, rank {RANK}, fit on {len(Ztr_fit)} pts')
    print(f'  {"sigma":>6} {"body":>9} {"PCA":>9} {"median":>9} '
          f'{"DM-kNN":>9} {"AE":>9}')
    for sigma in (0.02, 0.05, 0.10, 0.20):
        rng = np.random.default_rng(0)
        e_bod, e_pca, e_med, e_dm, e_ae = [], [], [], [], []
        for X in test:
            Z = emb(X)
            Y = Z + rng.normal(0, sigma, Z.shape)
            e_bod.append(np.linalg.norm(restore_knn(Ztr, Y) - Z, axis=1).mean())
            e_pca.append(np.linalg.norm(
                pca.inverse_transform(pca.transform(Y)) - Z, axis=1).mean())
            e_med.append(np.linalg.norm(
                median_filter(Y, size=(9, 1), mode='nearest') - Z,
                axis=1).mean())
            dm_Y = dm.transform(Y)
            nn = NearestNeighbors(n_neighbors=12).fit(dm.coords)
            dist, idx = nn.kneighbors(dm_Y)
            wgt = 1.0 / (dist + 1e-9)
            wgt /= wgt.sum(1, keepdims=True)
            rec_dm = np.einsum('nk,nkd->nd', wgt, dm.coords[idx])
            e_dm.append(np.linalg.norm(
                rec_dm - dm.transform(Z), axis=1).mean())
            with torch.no_grad():
                rec = ae(torch.tensor(Y, dtype=torch.float32)).numpy()
            e_ae.append(np.linalg.norm(rec - Z, axis=1).mean())
        r = [np.mean(e_bod), np.mean(e_pca), np.mean(e_med),
             np.mean(e_dm), np.mean(e_ae)]
        print(f'  {sigma:>6.2f} ' + ' '.join(f'{v:>9.3f}' for v in r))
    print('  (lower is better; AE = bottleneck-8 autoencoder reconstruction)')


if __name__ == '__main__':
    decode()
    denoise()
