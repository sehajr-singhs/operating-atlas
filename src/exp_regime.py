"""
Regime discovery: do the invariants recover the physics they were never told?

The robot testbed records a ground-truth regime for every sample -- slow or
fast, thermally nominal or derated, drive tripped -- and no model ever sees it.
The question is how much of that label is present in each candidate routing
coordinate system.

The comparison is held at equal dimension. A router given the raw 32-d state
has sixteen times as many numbers as one given (R, Pe), and would win on raw
predictive power without that meaning anything about geometry. So every
2-d candidate is scored against every other 2-d candidate:

    invariant   (R, Pe)
    naive       (Tr V, log det V)          basis-dependent stochastic features
    activity    (local speed, local var)   cheap non-geometric proxy
    random      a fixed random 2-d projection of the features
    pca2        the top two principal components of the raw state

with the full raw state reported alongside as a dimension-unconstrained
reference, not as a competitor.

Scores: adjusted mutual information between a 24-cell quantile binning of the
coordinate pair and the regime label, and the balanced accuracy of a k-nearest
-neighbour classifier fitted on the training split. Balanced accuracy is used
because the regimes are strongly unbalanced.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
CANDS = ['invariant', 'naive', 'activity', 'random', 'pca2']


def quantile_bins(Z, nb=5):
    """Joint quantile cell index, robust to the wildly different scales and
    tail shapes of the candidate coordinates."""
    out = np.zeros(len(Z), dtype=np.int64)
    for j in range(Z.shape[1]):
        q = np.quantile(Z[:, j], np.linspace(0, 1, nb + 1)[1:-1])
        out = out * nb + np.digitize(Z[:, j], q)
    return out


def score(Ztr, Ltr, Zte, Lte, seed=0, n_fit=40000, n_eval=40000):
    rng = np.random.RandomState(seed)
    a = rng.choice(len(Ztr), min(n_fit, len(Ztr)), replace=False)
    b = rng.choice(len(Zte), min(n_eval, len(Zte)), replace=False)
    sc = StandardScaler().fit(Ztr[a])
    knn = KNeighborsClassifier(n_neighbors=25, n_jobs=2).fit(sc.transform(Ztr[a]), Ltr[a])
    pred = knn.predict(sc.transform(Zte[b]))
    return dict(ami=float(adjusted_mutual_info_score(Lte[b], quantile_bins(Zte[b]))),
                bacc=float(balanced_accuracy_score(Lte[b], pred)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platform', default='ur5e')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    blob = torch.load(os.path.join(DATA, f'prep_{a.platform}_k{a.k}.pt'),
                      weights_only=False)
    if blob.get('L') is None:
        raise SystemExit('no regime labels in this prep artefact')
    Ltr = blob['L']['train'].numpy()
    Lte = blob['L']['test'].numpy()
    print(f'{a.platform}: train {len(Ltr)}, test {len(Lte)}, '
          f'regimes {np.bincount(Lte, minlength=5)}')

    coords = {}
    for c in CANDS[:-1]:
        coords[c] = (blob['R'][c]['train'].numpy(), blob['R'][c]['test'].numpy())
    p = PCA(2).fit(blob['S']['train'].numpy())
    coords['pca2'] = (p.transform(blob['S']['train'].numpy()),
                      p.transform(blob['S']['test'].numpy()))
    coords['raw (32-d, reference)'] = (blob['R']['raw']['train'].numpy(),
                                       blob['R']['raw']['test'].numpy())

    out = {'platform': a.platform, 'n_test': int(len(Lte)),
           'regime_counts': np.bincount(Lte, minlength=5).tolist(), 'scores': {}}
    for name, (Ztr, Zte) in coords.items():
        rows = [score(Ztr, Ltr, Zte, Lte, seed=s) for s in range(a.seeds)]
        agg = {m: (float(np.mean([r[m] for r in rows])),
                   float(np.std([r[m] for r in rows]))) for m in rows[0]}
        out['scores'][name] = agg
        print(f'  {name:24s} AMI {agg["ami"][0]:.3f}+-{agg["ami"][1]:.3f}   '
              f'balanced acc {agg["bacc"][0]:.3f}+-{agg["bacc"][1]:.3f}')

    with open(os.path.join(RES, f'regime_{a.platform}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'wrote results/regime_{a.platform}.json')


if __name__ == '__main__':
    main()
