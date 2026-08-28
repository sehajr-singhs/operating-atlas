"""
Transfer under sensor recalibration, with and without an accompanying shift in
the operating distribution.

The claim being tested is not that geometric routing is more accurate. It is
that the routing coordinates are *invariant*: relabelling the sensor
coordinates leaves them unchanged, so a router trained on one fleet keeps
making the same decisions on another. Raw-state routing has no such property.

Two target regimes, because they behave differently and the difference is the
point:

  matched  the target sessions have the same duty-cycle distribution as the
           source and differ only by calibration. A per-channel quantile match
           then inverts a channelwise warp outright, and raw routing should be
           fine. If the invariant arm does not win here, that is the expected
           result and is reported as such.

  shifted  the target is additionally restricted to a heavier part of the
           torque envelope. Now the marginals differ for two reasons at once,
           quantile matching cannot separate them, and the invariants should
           come into their own.

All adaptation is unsupervised: target labels are never used, by any arm.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from data import (load_pmsm, pmsm_splits, pmsm_arrays, Scramble, coral,
                  quantile_match, PMSM_STATE, PMSM_TARGET)
from pipeline import IOOFrontEnd
from models import random_projection
from runner import activity_features, train, evaluate, make_model

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
ARMS = ['mono', 'raw', 'invariant', 'raw+inv', 'naive', 'activity', 'random']


# ----------------------------------------------------------------------------

def make_blob(split_arrays, front, norm, targets):
    """Assemble the tensor dict the trainer expects from raw arrays."""
    xm, xs, ym, ys = norm
    d = {}
    for name, (X, Y, S, G) in split_arrays.items():
        d[name] = dict(X=X, Y=Y, S=S, G=G,
                       Xn=(X - xm) / xs, Yn=(Y - ym) / ys)
    R = {}
    for name in d:
        S, Xn, G = d[name]['S'], d[name]['Xn'], d[name]['G']
        inv = front.invariants(S)
        R.setdefault('invariant', {})[name] = inv
        R.setdefault('raw', {})[name] = S.clone()
        R.setdefault('raw+inv', {})[name] = torch.cat([S, inv], 1)
        R.setdefault('naive', {})[name] = front.naive_features(S)
        R.setdefault('random', {})[name] = random_projection(Xn, 2, seed=0)
        R.setdefault('activity', {})[name] = activity_features(S, G)
    return dict(Xn={k: d[k]['Xn'] for k in d}, Yn={k: d[k]['Yn'] for k in d},
                S={k: d[k]['S'] for k in d}, G={k: d[k]['G'] for k in d},
                R=R, ys=ys, targets=targets)


def standardise_R(R, stats):
    """Apply the *source* standardisation to target routing coordinates.

    This is the operative detail. The invariant arm reuses the source
    statistics unchanged, which is only legitimate because the quantities are
    invariant. Re-standardising them on the target would quietly launder away
    exactly the property under test.
    """
    out = {}
    for arm in R:
        m, s = stats[arm]
        out[arm] = {k: (v - m) / s for k, v in R[arm].items()}
    return out


def to_t(a):
    return torch.tensor(a, dtype=torch.float32)


def build(df, ids, warp=None):
    X, Y, S, G = pmsm_arrays(df, ids, warp=warp)
    return to_t(X), to_t(Y), to_t(S), torch.tensor(G.astype(np.int64))


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--warp-seeds', type=int, default=3)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-exact', type=int, default=8000)
    ap.add_argument('--subsample', type=int, default=4)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    cfg = dict(k=a.k, epochs=a.epochs, experts=6, width=64, expert_kind='mlp')

    print('loading PMSM ...')
    df = load_pmsm(max_profiles=24 if a.quick else None)
    sp = pmsm_splits(df, seed=0)

    sub = a.subsample
    tr = [x[::sub] for x in build(df, sp['train'])]
    va = [x[::sub] for x in build(df, sp['val'])]
    te = [x[::sub] for x in build(df, sp['test'])]

    xm, xs = tr[0].mean(0), tr[0].std(0) + 1e-8
    ym, ys = tr[1].mean(0), tr[1].std(0) + 1e-8
    norm = (xm, xs, ym, ys)

    print('fitting source front end ...')
    t0 = time.time()
    front_src = IOOFrontEnd(k=a.k, dt=0.5 * sub, seed=0).fit(
        tr[2], tr[3], chart_epochs=40, sde_epochs=40, n_exact=a.n_exact)
    print(f'  {time.time()-t0:.0f}s   fidelity {front_src.fidelity}')

    blob = make_blob({'train': tr, 'val': va, 'test': te}, front_src, norm, PMSM_TARGET)
    stats = {arm: (blob['R'][arm]['train'].mean(0),
                   blob['R'][arm]['train'].std(0) + 1e-8) for arm in blob['R']}
    blob['R'] = standardise_R(blob['R'], stats)

    print('training arms on the source fleet ...')
    models = {}
    for seed in range(a.seeds):
        for arm in ARMS:
            m = make_model(blob, arm, seed, cfg)
            m, _ = train(m, blob, arm, cfg, seed)
            models[(arm, seed)] = m
            print(f'  seed {seed} {arm:10s} clean test '
                  f'{evaluate(m, blob, arm, "test")["mse_mean"]:8.3f} K^2')

    results = {'config': vars(a), 'clean': {}, 'transfer': [],
               'fidelity_src': front_src.fidelity}
    for arm in ARMS:
        v = [evaluate(models[(arm, s)], blob, arm, 'test')['mse_mean']
             for s in range(a.seeds)]
        results['clean'][arm] = dict(mean=float(np.mean(v)), sd=float(np.std(v)))

    # ---- target conditions -------------------------------------------------
    Xs_src = tr[0].numpy()
    S_src = tr[2].numpy()
    torque_i = PMSM_STATE.index('torque')

    for dist in ['matched', 'shifted']:
        ids = sp['test']
        for mode in ['channelwise', 'mixing']:
            for wseed in range(a.warp_seeds):
                W = Scramble(len(PMSM_STATE), seed=wseed, mode=mode)
                Xw, Yw, Sw, Gw = [x[::sub] for x in build(df, ids, warp=W)]
                if dist == 'shifted':
                    thr = np.quantile(np.abs(te[2][:, torque_i].numpy()), 0.5)
                    keep = torch.tensor(np.abs(te[2][:, torque_i].numpy()) > thr)
                    Xw, Yw, Sw, Gw = Xw[keep], Yw[keep], Sw[keep], Gw[keep]
                if len(Xw) < 500:
                    continue

                t0 = time.time()
                front_tgt = IOOFrontEnd(k=a.k, dt=0.5 * sub, seed=0).fit(
                    Sw, Gw, chart_epochs=40, sde_epochs=40,
                    n_exact=a.n_exact, verbose=False)
                refit_secs = time.time() - t0

                for defence in ['standardise', 'coral', 'quantile']:
                    if defence == 'standardise':
                        Xa, Sa = Xw.numpy(), Sw.numpy()
                    elif defence == 'coral':
                        Xa, Sa = coral(Xw.numpy(), Xs_src), coral(Sw.numpy(), S_src)
                    else:
                        Xa = quantile_match(Xw.numpy(), Xs_src)
                        Sa = quantile_match(Sw.numpy(), S_src)
                    Xa, Sa = to_t(Xa), to_t(Sa)

                    tb = make_blob({'train': tr, 'val': va,
                                    'test': (Xa, Yw, Sa, Gw)}, front_src, norm,
                                   PMSM_TARGET)
                    # the invariant arms use the target-refit front end, which
                    # needs no labels; every other arm uses its defence
                    inv_t = front_tgt.invariants(Sw)
                    tb['R']['invariant']['test'] = inv_t
                    tb['R']['raw+inv']['test'] = torch.cat([Sa, inv_t], 1)
                    tb['R']['naive']['test'] = front_tgt.naive_features(Sw)
                    tb['R'] = standardise_R(tb['R'], stats)

                    row = dict(dist=dist, mode=mode, warp_seed=wseed,
                               defence=defence, n=int(len(Xa)),
                               refit_secs=refit_secs, arms={})
                    for arm in ARMS:
                        v = [evaluate(models[(arm, s)], tb, arm, 'test')['mse_mean']
                             for s in range(a.seeds)]
                        row['arms'][arm] = dict(mean=float(np.mean(v)),
                                                sd=float(np.std(v)))
                    # how well did the invariants actually survive the refit?
                    with torch.no_grad():
                        i_src = front_src.invariants(Sw)
                    for c, nm in enumerate(['R', 'Pe']):
                        r1 = np.argsort(np.argsort(inv_t[:, c].numpy()))
                        r0 = np.argsort(np.argsort(i_src[:, c].numpy()))
                        row[f'refit_rho_{nm}'] = float(np.corrcoef(r0, r1)[0, 1])
                    results['transfer'].append(row)
                    print(f'  {dist:8s} {mode:12s} w{wseed} {defence:12s} '
                          + '  '.join(f'{k}={row["arms"][k]["mean"]:7.2f}'
                                      for k in ['mono', 'raw', 'invariant']))

    with open(os.path.join(RES, 'pmsm_transfer.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nwrote results/pmsm_transfer.json')


if __name__ == '__main__':
    main()
