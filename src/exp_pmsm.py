"""
PMSM: in-distribution comparison of routing coordinates.

One expert bank, one training budget, one set of expert inputs. The arms differ
only in what the router is allowed to look at. Reported in kelvin^2 on the four
internal temperatures, over independent measurement sessions never seen in
training.

Usage:  python exp_pmsm.py [--quick] [--seeds 5] [--k 3]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from data import load_pmsm, pmsm_splits, pmsm_arrays, PMSM_TARGET
from models import OperatorAssembly, Monolith, count_params, match_monolith_width, random_projection
from pipeline import IOOFrontEnd

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


# ----------------------------------------------------------------------------

def build(args):
    df = load_pmsm(max_profiles=20 if args.quick else None)
    sp = pmsm_splits(df, seed=0)
    out = {}
    for name, ids in sp.items():
        X, Y, S, G = pmsm_arrays(df, ids)
        if args.quick:
            X, Y, S, G = X[::5], Y[::5], S[::5], G[::5]
        out[name] = dict(X=torch.tensor(X, dtype=torch.float32),
                         Y=torch.tensor(Y, dtype=torch.float32),
                         S=torch.tensor(S, dtype=torch.float32),
                         G=torch.tensor(G))
        print(f'  {name:5s} {len(ids):3d} sessions  {X.shape[0]:8d} samples')
    return out, sp


def standardize(d):
    xm, xs = d['train']['X'].mean(0), d['train']['X'].std(0) + 1e-8
    ym, ys = d['train']['Y'].mean(0), d['train']['Y'].std(0) + 1e-8
    for k in d:
        d[k]['Xn'] = (d[k]['X'] - xm) / xs
        d[k]['Yn'] = (d[k]['Y'] - ym) / ys
    return ys


def routing_coords(d, front, seed=0):
    """Every arm's router input, standardised on the training split."""
    R = {}
    for split in d:
        S, Xn = d[split]['S'], d[split]['Xn']
        inv = front.invariants(S)
        R.setdefault('invariant', {})[split] = inv
        R.setdefault('raw', {})[split] = S.clone()
        R.setdefault('raw+inv', {})[split] = torch.cat([S, inv], 1)
        R.setdefault('naive', {})[split] = front.naive_features(S)
        R.setdefault('random', {})[split] = random_projection(Xn, 2, seed=seed)
    # local activity control: speed and variance of the state within a session
    for split in d:
        S, G = d[split]['S'], d[split]['G']
        dS = torch.zeros_like(S)
        dS[1:] = S[1:] - S[:-1]
        dS[torch.cat([torch.tensor([True]), G[1:] != G[:-1]])] = 0
        sp = dS.norm(dim=-1, keepdim=True)
        w = 64
        ker = torch.ones(1, 1, w) / w
        sm = torch.nn.functional.conv1d(sp.T.unsqueeze(0), ker, padding=w // 2)[0].T[:len(sp)]
        var = torch.nn.functional.conv1d((sp ** 2).T.unsqueeze(0), ker, padding=w // 2)[0].T[:len(sp)] - sm ** 2
        R.setdefault('activity', {})[split] = torch.cat([sm, var], 1)

    for arm in R:
        m = R[arm]['train'].mean(0); s = R[arm]['train'].std(0) + 1e-8
        for split in R[arm]:
            R[arm][split] = (R[arm][split] - m) / s
    return R


# ----------------------------------------------------------------------------

def train_arm(model, d, rcoord, ys, epochs, bs=4096, lr=2e-3, patience=6, seed=0,
              verbose=False):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr, Ytr = d['train']['Xn'], d['train']['Yn']
    n = Xtr.shape[0]
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            r = rcoord['train'][j] if rcoord is not None else None
            pred = model(Xtr[j], r)
            loss = ((pred - Ytr[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        vm = evaluate(model, d, rcoord, ys, 'val')['mse_mean']
        if vm < best - 1e-4:
            best, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 5 == 0:
            print(f'      ep {ep:3d} val {vm:8.3f} K^2')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, d, rcoord, ys, split, bs=100000):
    model.eval()
    X, Y = d[split]['Xn'], d[split]['Yn']
    preds = []
    for i in range(0, X.shape[0], bs):
        r = rcoord[split][i:i + bs] if rcoord is not None else None
        preds.append(model(X[i:i + bs], r))
    P = torch.cat(preds)
    err = (P - Y) * ys                                  # back to kelvin
    mse = (err ** 2).mean(0)
    return dict(mse_mean=float(mse.mean()),
                mse_per_target={t: float(v) for t, v in zip(PMSM_TARGET, mse)},
                max_abs=float(err.abs().max()),
                p99=float(err.abs().flatten().quantile(0.99)))


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--experts', type=int, default=6)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--expert-kind', default='mlp')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    torch.set_default_dtype(torch.float32)

    print('loading PMSM ...')
    d, sp = build(args)
    ys = standardize(d)

    print(f'fitting IOO front end (unsupervised, k={args.k}) ...')
    t0 = time.time()
    front = IOOFrontEnd(k=args.k, dt=0.5, seed=0).fit(
        d['train']['S'], d['train']['G'],
        chart_epochs=20 if args.quick else 60,
        sde_epochs=20 if args.quick else 60,
        n_exact=4000 if args.quick else 20000)
    print(f'  front end fitted in {time.time()-t0:.0f}s')

    R = routing_coords(d, front, seed=0)
    d_in, d_out = d['train']['Xn'].shape[1], d['train']['Yn'].shape[1]

    ref = OperatorAssembly(d_in, d_out, 2, K=args.experts, width=args.width,
                           expert=args.expert_kind)
    target = count_params(ref)
    mono_w = match_monolith_width(d_in, d_out, target)
    print(f'assembly {target} params -> monolith width {mono_w} '
          f'({count_params(Monolith(d_in, d_out, mono_w))} params)')

    arms = ['mono', 'raw', 'invariant', 'raw+inv', 'naive', 'activity', 'random']
    results = {'config': vars(args), 'fidelity': front.fidelity,
               'n_train': int(d['train']['X'].shape[0]),
               'n_test': int(d['test']['X'].shape[0]),
               'sessions': {k: [int(x) for x in v] for k, v in sp.items()},
               'params': {}, 'runs': []}

    for seed in range(args.seeds):
        for arm in arms:
            t0 = time.time()
            if arm == 'mono':
                torch.manual_seed(seed)
                m = Monolith(d_in, d_out, mono_w)
                rc = None
            else:
                torch.manual_seed(seed)
                m = OperatorAssembly(d_in, d_out, R[arm]['train'].shape[1],
                                     K=args.experts, width=args.width,
                                     expert=args.expert_kind)
                rc = R[arm]
            results['params'][arm] = count_params(m)
            m = train_arm(m, d, rc, ys, epochs=args.epochs, seed=seed)
            te = evaluate(m, d, rc, ys, 'test')
            va = evaluate(m, d, rc, ys, 'val')
            results['runs'].append(dict(seed=seed, arm=arm, test=te, val=va,
                                        secs=time.time() - t0))
            print(f'  seed {seed} {arm:10s} test {te["mse_mean"]:8.3f} K^2  '
                  f'p99 {te["p99"]:6.2f} K  ({time.time()-t0:.0f}s)')

    tag = args.tag or ('quick' if args.quick else 'full')
    with open(os.path.join(RES, f'pmsm_{tag}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== summary (test MSE, K^2) ===')
    for arm in arms:
        v = [r['test']['mse_mean'] for r in results['runs'] if r['arm'] == arm]
        print(f'  {arm:10s} {np.mean(v):8.3f} +- {np.std(v):6.3f}   n={len(v)}')
    torch.save(front, os.path.join(RES, f'pmsm_front_{tag}.pt'))


if __name__ == '__main__':
    main()
