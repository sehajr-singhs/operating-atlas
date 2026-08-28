"""Collect per-job results into a table, with paired statistics against the
raw-state router (the arm the invariants actually have to beat)."""

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
ARMS = ['mono', 'raw', 'invariant', 'raw+inv', 'naive', 'activity', 'random']


def collect(dataset=None, tag='mlp', k=3):
    rows = []
    for f in glob.glob(os.path.join(RES, 'jobs', '*.json')):
        r = json.load(open(f))
        if dataset and r['dataset'] != dataset:
            continue
        rows.append(r)
    return rows


def table(dataset, tag='mlp', k=3, quiet=False):
    rows = [r for r in collect(dataset) if r['cfg'].get('k', 3) == k]
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault(r['arm'], {})[r['seed']] = r
    ref = by.get('raw', {})
    out = {'dataset': dataset, 'n_seeds': {}, 'arms': {}}
    if not quiet:
        print(f'\n=== {dataset} (k={k}) ===')
        print(f'{"arm":12s} {"test MSE":>18s} {"val MSE":>12s} {"params":>8s} '
              f'{"vs raw":>10s} {"p":>8s} {"gate H":>7s}')
    for a in ARMS:
        d = by.get(a, {})
        if not d:
            continue
        seeds = sorted(d)
        te = np.array([d[s]['test']['mse_mean'] for s in seeds])
        va = np.array([d[s]['val']['mse_mean'] for s in seeds])
        ent = [d[s].get('gate_entropy') for s in seeds if d[s].get('gate_entropy')]
        rel, p = '', ''
        common = sorted(set(seeds) & set(ref))
        if a != 'raw' and len(common) >= 3:
            x = np.array([d[s]['test']['mse_mean'] for s in common])
            y = np.array([ref[s]['test']['mse_mean'] for s in common])
            # paired over seeds: the same initialisation is used in both arms,
            # so the difference is the routing coordinate, not the draw
            t, pv = stats.ttest_rel(x, y)
            rel = f'{100*(x.mean()-y.mean())/y.mean():+.1f}%'
            p = f'{pv:.3f}'
        out['arms'][a] = dict(
            test_mean=float(te.mean()), test_sd=float(te.std(ddof=1) if len(te) > 1 else 0),
            val_mean=float(va.mean()), n=len(seeds),
            params=int(d[seeds[0]]['params']),
            gate_entropy=float(np.mean(ent)) if ent else None,
            rel_vs_raw=rel, p_vs_raw=p)
        if not quiet:
            print(f'{a:12s} {te.mean():10.4f} +-{te.std(ddof=1) if len(te)>1 else 0:6.4f} '
                  f'{va.mean():12.4f} {d[seeds[0]]["params"]:8d} {rel:>10s} {p:>8s} '
                  f'{(np.mean(ent) if ent else float("nan")):7.3f}  n={len(seeds)}')
    return out


if __name__ == '__main__':
    ds = sys.argv[1:] or ['cmapss', 'pmsm', 'ur5e', 'panda', 'iiwa14']
    all_out = {}
    for d in ds:
        t = table(d)
        if t:
            all_out[d] = t
    with open(os.path.join(RES, 'summary.json'), 'w') as f:
        json.dump(all_out, f, indent=2)
    print('\nwrote results/summary.json')
