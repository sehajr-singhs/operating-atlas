"""
Dataset preparation and single-arm training, shared by the local and Modal
entry points.

prep()    builds features, fits the unsupervised front end once, and caches
          every arm's routing coordinates so that all arms provably see the
          same numbers.
run_one() trains and evaluates a single (arm, seed).
"""

import json
import os
import time

import numpy as np
import torch

from models import (OperatorAssembly, Monolith, count_params,
                    match_monolith_width, random_projection)
from pipeline import IOOFrontEnd

ARMS = ['mono', 'raw', 'invariant', 'raw+inv', 'naive', 'activity', 'random']


# ----------------------------------------------------------------------------
# dataset construction
# ----------------------------------------------------------------------------

def build_pmsm(root, quick=False, subsample=1):
    import pandas as pd
    from data import pmsm_splits, pmsm_arrays, PMSM_TARGET
    csv = os.path.join(root, 'pmsm.csv')
    if not os.path.exists(csv):
        csv = os.path.join(root, 'pmsm', 'measures_v2.csv')
    df = pd.read_csv(csv)
    if quick:
        df = df[df.profile_id.isin(sorted(df.profile_id.unique())[:20])]
    sp = pmsm_splits(df, seed=0)
    out = {}
    for name, ids in sp.items():
        X, Y, S, G = pmsm_arrays(df, ids)
        if subsample > 1:
            X, Y, S, G = X[::subsample], Y[::subsample], S[::subsample], G[::subsample]
        out[name] = dict(X=torch.tensor(X, dtype=torch.float32),
                         Y=torch.tensor(Y, dtype=torch.float32),
                         S=torch.tensor(S, dtype=torch.float32),
                         G=torch.tensor(G.astype(np.int64)))
    return out, {k: [int(i) for i in v] for k, v in sp.items()}, PMSM_TARGET, 0.5


def build_cmapss(root, fd='FD004', quick=False):
    """RUL regression. State = operational settings + informative sensors."""
    import pandas as pd
    from data import CMAPSS_COLS, CMAPSS_SENSORS, CMAPSS_STATE
    p = os.path.join(root, 'cmapss')
    if not os.path.exists(os.path.join(p, f'train_{fd}.txt')):
        p = os.path.join(root, 'cmapss', 'CMaps')
    tr = pd.read_csv(os.path.join(p, f'train_{fd}.txt'), sep=r'\s+', header=None, names=CMAPSS_COLS)
    te = pd.read_csv(os.path.join(p, f'test_{fd}.txt'), sep=r'\s+', header=None, names=CMAPSS_COLS)
    rul = pd.read_csv(os.path.join(p, f'RUL_{fd}.txt'), sep=r'\s+', header=None, names=['RUL'])
    tr['RUL'] = (tr.groupby('unit').cycle.transform('max') - tr.cycle).clip(upper=125)
    te['RUL'] = ((te.groupby('unit').cycle.transform('max') - te.cycle)
                 + rul.RUL.to_numpy()[te.unit.to_numpy() - 1]).clip(upper=125)

    units = np.array(sorted(tr.unit.unique()))
    rng = np.random.RandomState(0); rng.shuffle(units)
    nval = max(1, int(0.2 * len(units)))
    val_u, tr_u = set(units[:nval]), set(units[nval:])

    def pack(d):
        S = d[CMAPSS_STATE].to_numpy(np.float32)
        # rolling window features over the last 30 cycles within a unit
        f = [S]
        g = d.groupby('unit')
        for w in (5, 15, 30):
            f.append(g[CMAPSS_STATE].transform(lambda s: s.rolling(w, min_periods=1).mean()).to_numpy(np.float32))
        f.append(g[CMAPSS_STATE].transform(lambda s: s.rolling(30, min_periods=1).std().fillna(0)).to_numpy(np.float32))
        f.append(d[['cycle']].to_numpy(np.float32))
        return (torch.tensor(np.concatenate(f, 1)),
                torch.tensor(d[['RUL']].to_numpy(np.float32)),
                torch.tensor(S), torch.tensor(d.unit.to_numpy()))

    out = {}
    out['train'] = dict(zip(['X', 'Y', 'S', 'G'], pack(tr[tr.unit.isin(tr_u)])))
    out['val'] = dict(zip(['X', 'Y', 'S', 'G'], pack(tr[tr.unit.isin(val_u)])))
    out['test'] = dict(zip(['X', 'Y', 'S', 'G'], pack(te)))
    return out, {}, ['RUL'], 1.0


def build_robot(root, platform='ur5e', quick=False, subsample=2):
    """Menagerie robot with injected joint thermals. Split by episode, which is
    the only leakage-free unit: each episode has its own payload, ambient
    temperature and excitation, and samples within one are strongly correlated.

    The ground-truth regime label L rides along but is never a model input; it
    is used only by the regime-discovery analysis.
    """
    d = np.load(os.path.join(root, f'robot_{platform}.npz'))
    X, Y, S, G, L = (d['X'], d['Y'], d['S'], d['G'], d['L'])
    if subsample > 1:
        X, Y, S, G, L = X[::subsample], Y[::subsample], S[::subsample], \
            G[::subsample], L[::subsample]
    eps = np.array(sorted(np.unique(G)))
    if quick:
        eps = eps[:12]
    rng = np.random.RandomState(0); rng.shuffle(eps)
    n = len(eps)
    parts = dict(train=eps[:int(.6 * n)], val=eps[int(.6 * n):int(.75 * n)],
                 test=eps[int(.75 * n):])
    out = {}
    for name, ids in parts.items():
        m = np.isin(G, ids)
        out[name] = dict(X=torch.tensor(X[m]), Y=torch.tensor(Y[m]),
                         S=torch.tensor(S[m]), G=torch.tensor(G[m]),
                         L=torch.tensor(L[m]))
    nj = (S.shape[1] - 2) // 5
    targets = ([f'dq{j}' for j in range(nj)] + [f'Tw{j}' for j in range(nj)])
    return out, {k: [int(i) for i in v] for k, v in parts.items()}, targets, 0.02 * subsample


BUILDERS = {'pmsm': build_pmsm, 'cmapss': build_cmapss,
            'ur5e': lambda r, **kw: build_robot(r, 'ur5e', **kw),
            'panda': lambda r, **kw: build_robot(r, 'panda', **kw),
            'iiwa14': lambda r, **kw: build_robot(r, 'iiwa14', **kw)}


# ----------------------------------------------------------------------------

def activity_features(S, G, w=64):
    """Local speed and local variance: the cheap non-geometric control."""
    dS = torch.zeros_like(S)
    dS[1:] = S[1:] - S[:-1]
    brk = torch.cat([torch.tensor([True]), G[1:] != G[:-1]])
    dS[brk] = 0
    sp = dS.norm(dim=-1, keepdim=True)
    ker = torch.ones(1, 1, w) / w
    m = torch.nn.functional.conv1d(sp.T.unsqueeze(0), ker, padding=w // 2)[0].T[:len(sp)]
    m2 = torch.nn.functional.conv1d((sp ** 2).T.unsqueeze(0), ker, padding=w // 2)[0].T[:len(sp)]
    return torch.cat([m, (m2 - m ** 2).clamp_min(0)], 1)


def prep(root, dataset='pmsm', k=3, n_exact=20000, quick=False, out=None,
         subsample=1, verbose=True):
    torch.set_default_dtype(torch.float32)
    kw = dict(quick=quick)
    if dataset in ('pmsm', 'ur5e', 'panda', 'iiwa14'):
        kw['subsample'] = subsample
    d, sessions, targets, dt = BUILDERS[dataset](root, **kw)
    for s in d:
        if verbose:
            print(f'  {s:5s} {d[s]["X"].shape[0]:8d} samples, {d[s]["X"].shape[1]} feats')

    xm, xs = d['train']['X'].mean(0), d['train']['X'].std(0) + 1e-8
    ym, ys = d['train']['Y'].mean(0), d['train']['Y'].std(0) + 1e-8
    for s in d:
        d[s]['Xn'] = (d[s]['X'] - xm) / xs
        d[s]['Yn'] = (d[s]['Y'] - ym) / ys

    t0 = time.time()
    front = IOOFrontEnd(k=k, dt=dt, seed=0).fit(
        d['train']['S'], d['train']['G'],
        chart_epochs=20 if quick else 60, sde_epochs=20 if quick else 60,
        n_exact=n_exact, verbose=verbose)
    front_secs = time.time() - t0

    R = {}
    for s in d:
        S, Xn, G = d[s]['S'], d[s]['Xn'], d[s]['G']
        inv = front.invariants(S)
        R.setdefault('invariant', {})[s] = inv
        R.setdefault('raw', {})[s] = S.clone()
        R.setdefault('raw+inv', {})[s] = torch.cat([S, inv], 1)
        R.setdefault('naive', {})[s] = front.naive_features(S)
        R.setdefault('random', {})[s] = random_projection(Xn, 2, seed=0)
        R.setdefault('activity', {})[s] = activity_features(S, G)
    for arm in R:
        m = R[arm]['train'].mean(0); sd = R[arm]['train'].std(0) + 1e-8
        for s in R[arm]:
            R[arm][s] = (R[arm][s] - m) / sd

    blob = dict(
        Xn={s: d[s]['Xn'] for s in d}, Yn={s: d[s]['Yn'] for s in d},
        S={s: d[s]['S'] for s in d}, G={s: d[s]['G'] for s in d},
        R=R, ys=ys, targets=targets, sessions=sessions, dt=dt,
        fidelity=front.fidelity, front_secs=front_secs, k=k,
        exact_invariants=front._F, exact_Z=front._Zs,
        L={s: d[s]['L'] for s in d} if 'L' in d['train'] else None)
    if out:
        torch.save(blob, out)
        torch.save(front, out.replace('.pt', '_front.pt'))
        if verbose:
            print(f'  wrote {out}  (front end {front_secs:.0f}s)')
    return blob


# ----------------------------------------------------------------------------

def evaluate(model, blob, arm, split, bs=200000):
    model.eval()
    X, Y, ys = blob['Xn'][split], blob['Yn'][split], blob['ys']
    with torch.no_grad():
        P = torch.cat([model(X[i:i + bs],
                             None if arm == 'mono' else blob['R'][arm][split][i:i + bs])
                       for i in range(0, X.shape[0], bs)])
    err = (P - Y) * ys
    mse = (err ** 2).mean(0)
    return dict(mse_mean=float(mse.mean()),
                per_target={t: float(v) for t, v in zip(blob['targets'], mse)},
                rmse=float(mse.mean().sqrt()),
                p99=float(err.abs().flatten().quantile(0.99)),
                max_abs=float(err.abs().max()))


def make_model(blob, arm, seed, cfg):
    d_in = blob['Xn']['train'].shape[1]
    d_out = blob['Yn']['train'].shape[1]
    torch.manual_seed(seed)
    if arm == 'mono':
        ref = OperatorAssembly(d_in, d_out, 2, K=cfg['experts'], width=cfg['width'],
                               expert=cfg['expert_kind'])
        w = match_monolith_width(d_in, d_out, count_params(ref))
        torch.manual_seed(seed)
        return Monolith(d_in, d_out, w)
    return OperatorAssembly(d_in, d_out, blob['R'][arm]['train'].shape[1],
                            K=cfg['experts'], width=cfg['width'],
                            tau=cfg.get('tau', 1.0), expert=cfg['expert_kind'])


def train(model, blob, arm, cfg, seed, verbose=False):
    torch.manual_seed(seed + 1000)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 2e-3))
    epochs = cfg['epochs']
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X, Y = blob['Xn']['train'], blob['Yn']['train']
    Rtr = None if arm == 'mono' else blob['R'][arm]['train']
    n, bs = X.shape[0], cfg.get('bs', 4096)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            loss = ((model(X[j], None if Rtr is None else Rtr[j]) - Y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        v = evaluate(model, blob, arm, 'val')['mse_mean']
        if v < best - 1e-4:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.get('patience', 6):
                break
        if verbose and ep % 5 == 0:
            print(f'    ep {ep:3d} val {v:9.4f}')
    if state:
        model.load_state_dict(state)
    return model, best


def run_one(root, dataset, arm, seed, cfg, blob=None):
    torch.set_default_dtype(torch.float32)
    if blob is None:
        blob = torch.load(os.path.join(root, f'prep_{dataset}_k{cfg["k"]}.pt'),
                          weights_only=False)
    t0 = time.time()
    m = make_model(blob, arm, seed, cfg)
    m, best_val = train(m, blob, arm, cfg, seed)
    r = dict(dataset=dataset, arm=arm, seed=seed, params=count_params(m),
             val=evaluate(m, blob, arm, 'val'), test=evaluate(m, blob, arm, 'test'),
             secs=time.time() - t0, cfg=cfg)
    if arm != 'mono':
        with torch.no_grad():
            p = m.gate(blob['R'][arm]['test'][:200000])
            r['gate_entropy'] = float(-(p * (p + 1e-12).log()).sum(-1).mean())
            r['gate_max'] = float(p.max(-1).values.mean())
            r['expert_usage'] = p.mean(0).tolist()
    return r
