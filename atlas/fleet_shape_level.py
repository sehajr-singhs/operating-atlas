"""
Controlled experiment: what makes sibling machines retrievable by an
invariant representation?

The reviewer-facing limitation of the atlas framework is that it identifies
operating EPISODES of one machine (32.5% on the motor bench) but not distinct
SIBLING machines in a fleet (0% for chart geometry, chance-level). The fleet
identities in the default generation vary payload, thermal resistances,
winding resistance, damping, servo gain and skew all at once -- and payload is
an ABSOLUTE LEVEL property, exactly the information the invariant description
throws away.

This experiment isolates the two kinds of identity:

    level-only fleet   payload varies 0-5 kg, everything else neutral.
                       The invariant atlas SHOULD fail here: differences are
                       magnitudes, which ranks discard by construction.

    shape-only fleet   payload fixed at 2.5 kg, the six relational parameters
                       (r_wh, r_ha, k_cu, damp, gain, skew) vary over their
                       full ranges. Differences are in coupling, lag, loop
                       shape, curvature -- shape properties. If the atlas
                       carries any machine identity at all, it should work
                       HERE or nowhere.

    all fleet          everything varies (the default generation).

Same units, episodes, workload and chart protocol in all three, so the only
thing that changes is which physical parameters carry the identity.
"""

import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import robot_units as ru
import ioo_chart as ic
from scipy.stats import skew, kurtosis
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUT = os.path.expanduser('~/kaggle_kernel/out/fleet_expt')
PLATFORM = os.environ.get('PLATFORM', 'ur5e')
N_UNITS = int(os.environ.get('N_UNITS', 40))
N_EP = int(os.environ.get('N_EP', 6))
SECONDS = float(os.environ.get('SECONDS', 45.0))
WORKERS = int(os.environ.get('WORKERS', 4))
MAXN = int(os.environ.get('MAXN', 6000))
IDENT = ['payload', 'r_wh', 'r_ha', 'k_cu', 'damp', 'gain', 'skew']

# neutral / full ranges (mirror ru.IDENTITY)
NEUTRAL = np.array([2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
RANGES = np.array([[0.0, 5.0], [0.7, 1.6], [0.7, 2.0], [0.8, 1.8],
                   [0.5, 2.5], [0.88, 1.12], [0.0, 1.0]])

CONFIGS = {
    # (mask, name): True = parameter varies, False = held at neutral
    'level_only': (np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool), 'level_only'),
    'shape_only': (np.array([0, 1, 1, 1, 1, 1, 1], dtype=bool), 'shape_only'),
    'all':        (np.array([1, 1, 1, 1, 1, 1, 1], dtype=bool), 'all'),
}


def sample_identity(mask, rng):
    v = NEUTRAL.copy()
    v[mask] = RANGES[mask, 0] + (RANGES[mask, 1] - RANGES[mask, 0]) * rng.random(mask.sum())
    return v


def _one(args):
    ui, ident, ep, seed = args
    wrng = np.random.RandomState(500000 + 977 * ui + ep)
    duty = float(wrng.uniform(0.25, 1.0))
    t_env = float(wrng.uniform(18.0, 34.0))
    try:
        rows, labs = ru.rollout_unit(PLATFORM, ident, unit_seed=1000 + ui,
                                     ep_seed=100000 + 137 * ui + ep,
                                     seconds=SECONDS, telemetry_hz=50.0,
                                     duty=duty, t_env=t_env)
        return (ui, ep, rows, None)
    except Exception as e:
        return (ui, ep, None, str(e))


def build_fleet(mask, name, seed=0):
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(seed)
    idents = np.array([sample_identity(mask, rng) for _ in range(N_UNITS)])
    jobs = [(ui, idents[ui], ep, seed) for ui in range(N_UNITS)
            for ep in range(N_EP)]
    t0 = time.time()
    eps = {}
    fails = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (ui, ep, rows, err) in enumerate(ex.map(_one, jobs, chunksize=1)):
            if rows is None:
                fails += 1
                continue
            eps.setdefault(ui, []).append((ep, rows))
            if (k + 1) % 60 == 0:
                print(f'    {name} {k+1}/{len(jobs)} eps '
                      f'[{time.time()-t0:.0f}s, {fails} dropped]', flush=True)
    print(f'  {name}: {len(eps)} units x {N_EP} eps x {SECONDS:.0f}s, '
          f'{fails} dropped, {time.time()-t0:.0f}s', flush=True)
    return idents, eps


def stack(eps_u, keys, maxn=MAXN):
    X = np.concatenate([eps_u[k] for k in keys]).astype(np.float64)
    return X[::max(1, len(X) // maxn)][:maxn]


def marginal(X):
    return np.nan_to_num(np.concatenate(
        [X.mean(0), X.std(0), skew(X, 0), kurtosis(X, 0)]))


def retrieval(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = An @ Bn.T
    r = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1)
    return float((r == 0).mean()), float((1 - r / (len(S) - 1)).mean())


def decode(F, Y):
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=int(min(24, F.shape[0] // 4, F.shape[1]))),
                         RidgeCV(alphas=np.logspace(-2, 5, 30)))
    cv = KFold(5, shuffle=True, random_state=0)
    out = {}
    for j, nm in enumerate(IDENT):
        y = Y[:, j]
        if y.std() < 1e-9:
            out[nm] = float('nan'); continue
        p = cross_val_predict(pipe, F, y, cv=cv)
        out[nm] = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return out


def evaluate(idents, eps, name):
    t0 = time.time()
    units = sorted(eps)
    epsd = {u: {e: r for e, r in eps[u]} for u in units}
    rng = np.random.default_rng(0)
    order = list(rng.permutation(units))
    train_u, test_u = order[10:], order[:10]

    halves = {}
    for u in units:
        ks = sorted(epsd[u])
        h = len(ks) // 2
        halves[u] = (stack(epsd[u], ks[:h]), stack(epsd[u], ks[h:]))
    print(f'    eval: stacks {time.time()-t0:.0f}s', flush=True)

    t0 = time.time()
    chart = ic.OperatorChart(n_landmarks=120, k=48).fit_class(
        [halves[u][0] for u in train_u])
    print(f'    eval: chart fit {time.time()-t0:.0f}s', flush=True)
    t0 = time.time()
    ioos_a = {u: chart.ioo(halves[u][0]) for u in units}
    print(f'    eval: ioos_a {time.time()-t0:.0f}s', flush=True)
    t0 = time.time()
    chart.set_core([ioos_a[u] for u in train_u], min_frac=0.9)
    ioos_b = {u: chart.ioo(halves[u][1]) for u in units}
    print(f'    eval: ioos_b {time.time()-t0:.0f}s', flush=True)

    A = np.stack([chart.descriptor(ioos_a[u]) for u in test_u])
    B = np.stack([chart.descriptor(ioos_b[u]) for u in test_u])
    Am = np.stack([marginal(halves[u][0]) for u in test_u])
    Bm = np.stack([marginal(halves[u][1]) for u in test_u])

    tg, prg = retrieval(A, B)
    tm, prm = retrieval(Am, Bm)
    chance = 100.0 / len(test_u)
    print(f'\n  [{name}]  retrieval on {len(test_u)} held-out siblings '
          f'(chance {chance:.1f}%)')
    print(f'    geometry   top-1 {100*tg:5.1f} %   pct-rank {100*prg:5.1f} %')
    print(f'    marginals  top-1 {100*tm:5.1f} %   pct-rank {100*prm:5.1f} %')

    print(f'  [{name}]  decoding TRUE physical parameters (all {len(units)} units)')
    Y = np.stack([idents[u] for u in units])
    Fg = np.stack([chart.descriptor(ioos_a[u]) for u in units])
    Fm = np.stack([marginal(halves[u][0]) for u in units])
    dg = decode(Fg, Y)
    dm = decode(Fm, Y)
    print('    geometry : ' + '  '.join(f'{k}={v:+.2f}' for k, v in dg.items()))
    print('    marginal: ' + '  '.join(f'{k}={v:+.2f}' for k, v in dm.items()))
    return dict(name=name, top1_geom=tg, prank_geom=prg, top1_marg=tm,
                prank_marg=prm, decode_geom=dg, decode_marg=dm, chance=chance)


def main():
    os.makedirs(OUT, exist_ok=True)
    which = os.environ.get('FLEETS', 'level_only,shape_only,all').split(',')
    summary = {}
    for name in which:
        mask = CONFIGS[name][0]
        t0 = time.time()
        print(f'\n=== fleet: {name} ===', flush=True)
        idents, eps = build_fleet(mask, name)
        print(f'  build {time.time()-t0:.0f}s', flush=True)
        t0 = time.time()
        summary[name] = evaluate(idents, eps, name)
        print(f'  eval {time.time()-t0:.0f}s', flush=True)
    print('\n=== SUMMARY ===')
    for name, s in summary.items():
        print(f'  {name:<12} geometry top-1 {100*s["top1_geom"]:5.1f} %  '
              f'marginals top-1 {100*s["top1_marg"]:5.1f} %  '
              f'(chance {100*s["chance"]:.1f} %)')
    import json
    with open(os.path.join(OUT, f'shape_level_{PLATFORM}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('wrote', os.path.join(OUT, f'shape_level_{PLATFORM}.json'))


if __name__ == '__main__':
    main()
