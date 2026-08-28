"""
The operating body as a physical prior.

Identification and decoding are diagnostics. The use that matters for control is
different: a body says which states the machine is ALLOWED to occupy, so a
reading that falls off it is partly measurement error and can be corrected
toward the body. No per-channel filter can express that, because the constraint
is a relation between channels and a per-channel filter only knows about one
channel at a time.

The test is deliberately blunt. Corrupt held-out telemetry with noise, then
restore it three ways:

    per-channel median filter   the standard fix, knows nothing about coupling
    linear subspace (PCA)       knows about linear coupling, the fair baseline
    the operating body          nearest neighbours on the body built from OTHER
                                runs of the same machine class

The body is built from training runs only, so it cannot memorise the test data.
If it does not beat a linear subspace of matched rank, the nonlinearity of the
body is not doing any work and the whole geometric apparatus is decoration.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manifold_local as ml
from sklearn.decomposition import PCA            # noqa: E402
from sklearn.neighbors import NearestNeighbors   # noqa: E402
from scipy.ndimage import median_filter          # noqa: E402

P = os.path.expanduser('~/probe/cnc')
RANK = int(os.environ.get('RANK', 8))     # matched capacity for PCA and the body


def load():
    fs_ = sorted(glob.glob(os.path.join(P, 'experiment_*.csv')))
    d0 = pd.read_csv(fs_[0])
    cols = [c for c in d0.columns if c != 'Machining_Process'
            and pd.to_numeric(d0[c], errors='coerce').notna().mean() > 0.9]
    runs = []
    for f in fs_:
        X = pd.read_csv(f)[cols].to_numpy(float)
        runs.append(X[np.isfinite(X).all(1)])
    return runs, cols


def restore_body(train_Z, Y, k=12):
    """Pull each reading toward the body traced by the training runs."""
    nn = NearestNeighbors(n_neighbors=k).fit(train_Z)
    dist, idx = nn.kneighbors(Y)
    w = 1.0 / (dist + 1e-9)
    w /= w.sum(1, keepdims=True)
    return np.einsum('nk,nkd->nd', w, train_Z[idx])


def main():
    runs, cols = load()
    print(f'{len(runs)} runs, {len(cols)} channels')

    # a shared monotone embedding fitted on the TRAINING runs only
    train, test = runs[:12], runs[12:]
    pool = np.concatenate(train)
    grid = np.linspace(0, 1, 512)
    knots = [np.quantile(pool[:, j], grid) for j in range(pool.shape[1])]

    def emb(X):
        return np.stack([np.interp(X[:, j], knots[j], grid)
                         for j in range(X.shape[1])], 1)

    Ztr = emb(pool)
    w = np.sqrt(np.maximum(ml.channel_snr(pool) - 0.02, 0.0))
    live = w > 1e-6
    Ztr = Ztr[:, live] * w[live]
    print(f'body from {len(train)} training runs: {Ztr.shape[0]} points, '
          f'{Ztr.shape[1]} live channels, '
          f'intrinsic dim {ml.intrinsic_dim_twonn(Ztr[::4]):.2f}')

    pca = PCA(n_components=RANK).fit(Ztr)

    print(f'\nrestoring corrupted held-out telemetry ({len(test)} runs, '
          f'PCA rank {RANK})')
    print(f'  {"noise":>7}  {"corrupted":>10}{"median filt":>13}'
          f'{"PCA":>10}{"body":>10}   {"body vs PCA":>12}')
    for sigma in (0.02, 0.05, 0.10, 0.20):
        rows = []
        rng = np.random.default_rng(0)
        for X in test:
            Z = emb(X)[:, live] * w[live]
            Y = Z + rng.normal(0, sigma, Z.shape)
            e_raw = np.linalg.norm(Y - Z, axis=1).mean()
            e_med = np.linalg.norm(median_filter(Y, size=(9, 1), mode='nearest')
                                   - Z, axis=1).mean()
            e_pca = np.linalg.norm(pca.inverse_transform(pca.transform(Y)) - Z,
                                   axis=1).mean()
            e_bod = np.linalg.norm(restore_body(Ztr, Y) - Z, axis=1).mean()
            rows.append((e_raw, e_med, e_pca, e_bod))
        r = np.array(rows).mean(0)
        gain = 100 * (1 - r[3] / max(r[2], 1e-12))
        print(f'  {sigma:>7.2f}  {r[0]:>10.4f}{r[1]:>13.4f}{r[2]:>10.4f}'
              f'{r[3]:>10.4f}   {gain:>+11.1f}%')

    print('\n  (lower is better; the body is built from training runs only,'
          '\n   so it cannot memorise the test telemetry)')


if __name__ == '__main__':
    main()
