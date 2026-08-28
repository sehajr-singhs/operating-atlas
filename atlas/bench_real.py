"""
The real-hardware benchmark.

Four experiments, chosen to match what public industrial data can actually
support rather than what would be convenient to claim.

  R1  CLASS SEPARATION across every real system available, using the
      atom-quantile profile so systems with 4, 7, 11 and 17 channels are
      commensurable. Records are strided to a common length first, because two
      atoms track acquisition settings rather than physics.

  R2  REAL DRIFT. A single industrial gas turbine logged hourly for five years.
      Nothing is warped by us; this is the asset ageing, the sensors ageing, and
      five years of maintenance. We ask how far the atlas moves between years
      against how far per-channel statistics move, and we ask whether a
      classifier can tell which YEAR a segment came from. A representation that
      tracks the machine's relational physics should be hard to date; one that
      tracks absolute levels should be easy to date. Being hard to date is the
      property we want, and it is the honest real-world analogue of the
      synthetic recalibration test.

  R3  FAULT DIAGNOSIS on the hydraulic rig, which ships ground-truth condition
      labels for four components. This replaces the forward-prediction task that
      unit conditioning failed: it asks whether the relational description
      supports a real diagnostic decision, which is what a condition-monitoring
      system is actually for.

  R4  RECALIBRATION across every real system at once, rather than on one bench.

Baselines throughout: per-channel marginal statistics, and the atom vocabulary
reduced to its first entry, a Spearman correlation matrix.
"""

import json
import os
import sys
import time
import numpy as np

IN = os.environ.get('IOO_IN', '/kaggle/input')
OUT = os.environ.get('IOO_OUT', '/kaggle/working')
FIXED_N = int(os.environ.get("FIXED_N", 1000))
SEED = int(os.environ.get('SEED', 0))


def _find(marker, roots=(IN, '/kaggle/input')):
    for r in roots:
        if not os.path.isdir(r):
            continue
        for root, dirs, files in os.walk(r):
            if marker in files or marker in dirs:
                return root
    raise FileNotFoundError(marker)


sys.path.insert(0, _find('atoms.py'))
import atoms                                      # noqa: E402
# the assets dataset also ships real_systems.py (with different loader
# defaults); this kernel's copy must win, so re-insert this module's own
# directory at the head of the path before importing it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_systems as rsys                       # noqa: E402
from scipy.stats import skew, kurtosis            # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline        # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

QS = np.array([0.02, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.98])
NA = len(atoms.ATOM_NAMES)
LOG = open(os.path.join(OUT, 'bench_real.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def stride_to(X, n):
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(1)]
    if len(X) < n:
        return None
    return X[::max(1, len(X) // n)][:n]


def marginal(X):
    """Per-channel statistics. Length is 4d, so this is only comparable between
    records of the SAME system."""
    return np.nan_to_num(np.concatenate(
        [X.mean(0), X.std(0), skew(X, 0), kurtosis(X, 0)]))


def marginal_profile(X):
    """The channel-count-free form of the marginal baseline.

    Comparing systems with 4, 7, 11, 17 and 24 channels means the baseline has
    to be made commensurable the same way the atlas is, by taking quantiles
    across channels instead of concatenating them. Concatenating raw per-channel
    statistics gives vectors of different length per system, which cannot be fed
    to one classifier at all, and quietly padding them to a common width would
    hand the classifier the channel count as a free label.
    """
    Z = np.nan_to_num(np.stack([X.mean(0), X.std(0), skew(X, 0), kurtosis(X, 0)]))
    # standardise each statistic across channels so scale does not dominate
    Zs = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-9)
    return np.concatenate([np.quantile(z, QS) for z in Zs])


def profile(X, warp_rng=None):
    """Atom-quantile class signature, plus the raw atlas for per-system work."""
    if warp_rng is not None:
        X = atoms.warp_channels(X, warp_rng)
    keep = X.std(0) > 1e-10
    X = X[:, keep]
    if X.shape[1] < 3:
        return None, None
    A = atoms.atlas_unit(X, n_cells=1, min_cell=100)[0]
    prof = np.empty((NA, len(QS)))
    for ai in range(NA):
        v = A[:, ai][np.isfinite(A[:, ai])]
        prof[ai] = np.quantile(v, QS) if len(v) >= 5 else np.nan
    return prof.reshape(-1), A


def cv_acc(F, y, tag, folds=5, quiet=False):
    F = np.nan_to_num(np.asarray(F))
    y = np.asarray(y)
    keep = F.std(0) > 1e-9
    F = F[:, keep]
    import pandas as pd
    nf = int(min(folds, pd.Series(y).value_counts().min()))
    if nf < 2 or F.shape[1] < 2 or len(set(y)) < 2:
        if not quiet:
            log(f'    {tag:<34} skipped')
        return float('nan'), float('nan')
    if y.dtype.kind == 'f':
        y = np.round(y).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(nf, shuffle=True, random_state=SEED)
    s = cross_val_score(clf, F, y, cv=cv)
    if not quiet:
        maj = pd.Series(y).value_counts().max() / len(y)
        log(f'    {tag:<34} {s.mean()*100:5.1f} +- {s.std()*100:4.1f} %   '
            f'(majority {maj*100:.1f} %, {len(set(y))} classes, n={len(y)})')
    return float(s.mean()), float(s.std())


def boot_ci(F, y, B=200, folds=5, seed=0, frac=0.7):
    """Subsampling 95% CI on the CV accuracy: each resample draws a 70%
    subsample WITHOUT replacement, so no record appears in both train and test
    of the same fold (resampling with replacement inflates CV accuracy by
    leaking near-duplicates into the test fold, which showed up as CIs far
    above the point estimate on the two-class fault problems)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    scores = []
    m = int(max(frac * n, 2 * folds))
    for b in range(B):
        idx = rng.choice(n, m, replace=False)
        s, _ = cv_acc(F[idx], y[idx], f'boot {b}', folds=folds, quiet=True)
        if np.isfinite(s):
            scores.append(s)
    if len(scores) < 50:
        return float('nan'), float('nan')
    scores = np.sort(scores)
    return float(scores[int(0.025 * len(scores))]), \
        float(scores[int(0.975 * len(scores))])


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    R = {'fixed_n': FIXED_N}

    # ---- load every real system -----------------------------------------
    recs = {}
    for name, fn in rsys.REAL_LOADERS.items():
        try:
            root = _find_data(name)
            got = fn(root)
            recs[name] = got
            d = got[0][1].shape[1] if got else 0
            log(f'  {name:<12} {len(got):4d} records, {d:2d} channels  '
                f'[{time.time()-t0:.0f}s]')
        except Exception as e:
            log(f'  {name:<12} SKIPPED ({type(e).__name__}: {e})')
    if not recs:
        log('no real systems loaded'); return

    # ---- build profiles --------------------------------------------------
    P, L, M, Mraw, meta = [], [], [], [], []
    Pw, Mw, Mraww = [], [], []
    drops = {}
    for name, got in recs.items():
        for k, (lab, X, mt) in enumerate(got):
            Xs = stride_to(X, FIXED_N)
            if Xs is None:
                d = drops.setdefault(name, [])
                if len(d) < 3:
                    d.append(f'rec{k}: stride_to None (len {len(X)})')
                continue
            seed = 5000 + len(P)
            Xw = atoms.warp_channels(Xs, np.random.default_rng(seed))
            p, A = profile(Xs)
            pw, _ = profile(Xs, warp_rng=np.random.default_rng(seed))
            if p is None or pw is None:
                d = drops.setdefault(name, [])
                if len(d) < 3:
                    nk = int((Xs.std(0) > 1e-10).sum())
                    d.append(f'rec{k}: profile None (channels {nk})')
                continue
            if not (np.isfinite(p).all() and np.isfinite(pw).all()):
                d = drops.setdefault(name, [])
                if len(d) < 3:
                    d.append(f'rec{k}: non-finite atoms '
                             f'{int((~np.isfinite(p)).sum())}/'
                             f'{int((~np.isfinite(pw)).sum())}')
                continue
            P.append(p); Pw.append(pw); L.append(lab)
            M.append(marginal_profile(Xs)); Mw.append(marginal_profile(Xw))
            Mraw.append(marginal(Xs)); Mraww.append(marginal(Xw))
            meta.append(dict(system=name, **{k2: v for k2, v in mt.items()
                                             if k2 != 'channels'}))
    P, Pw = np.array(P), np.array(Pw)
    M, Mw = np.array(M), np.array(Mw)
    L = np.array(L)
    # raw per-channel marginals stay ragged across systems and are only used
    # inside a single system, where the channel count is fixed
    Mraw = np.array(Mraw, dtype=object)
    Mraww = np.array(Mraww, dtype=object)
    log(f'\nprofiles {P.shape} across {len(set(L))} real systems '
        f'[{time.time()-t0:.0f}s]')
    for c in sorted(set(L)):
        log(f'  {c:<14} n={int((L == c).sum())}')
    # any system that lost ALL its records gets a diagnostic
    for name, got in recs.items():
        if name not in set(L) and got:
            log(f'  !! {name}: all {len(got)} records dropped: '
                + '; '.join(drops.get(name, ['unknown'])))

    # ---- R1 class separation --------------------------------------------
    log('\n=== R1  class separation across real systems ===')
    R['R1'] = {}
    r1_cfg = [('atlas', P, 'atom-quantile profile'),
              ('marginal', M, 'marginal-quantile profile')]
    rho_block = P.reshape(len(P), NA, len(QS))[:, 0, :]
    r1_cfg.append(('rho', rho_block, 'correlation quantiles only'))
    rate_free = [i for i, nm in enumerate(atoms.ATOM_NAMES)
                 if nm not in ('tau', 'fill')]
    Pr = P.reshape(len(P), NA, len(QS))[:, rate_free, :].reshape(len(P), -1)
    r1_cfg.append(('no_tau_fill', Pr, 'without tau and fill'))
    for key, F, tag in r1_cfg:
        R['R1'][key] = cv_acc(F, L, tag)
        lo, hi = boot_ci(F, L, B=100)
        R['R1'][key + '_ci'] = [lo, hi]
        log(f'    {key:<12} 95% CI [{lo*100:.1f}, {hi*100:.1f}]%')
    log(f'    R1 across {len(set(L))} real systems, n={len(L)} records')

    # ---- R4 recalibration ------------------------------------------------
    log('\n=== R4  recalibration, every real system at once ===')
    R['R4'] = {}
    R['R4']['atlas_warped'] = cv_acc(Pw, L, 'atlas, channels recalibrated')
    R['R4']['marginal_warped'] = cv_acc(Mw, L, 'marginal profile, recalibrated')
    inv = [i for i, nm in enumerate(atoms.ATOM_NAMES) if nm in atoms.WARP_INVARIANT]
    Pi = P.reshape(len(P), NA, len(QS))[:, inv, :]
    Pwi = Pw.reshape(len(Pw), NA, len(QS))[:, inv, :]
    shift = float(np.nanmax(np.abs(Pi - Pwi)))
    R['R4']['max_invariant_shift'] = shift
    log(f'    max |shift| over the eight invariant atoms: {shift:.2e}')

    # ---- R2 real five-year drift ----------------------------------------
    log('\n=== R2  real drift, one gas turbine over five years ===')
    if 'gasturbine' in recs:
        gt_idx = [i for i, m in enumerate(meta) if m['system'] == 'gasturbine']
        yr = np.array([meta[i].get('year', -1) for i in gt_idx])
        # inside one system the channel count is fixed, so the raw per-channel
        # marginals are available and are the stronger baseline
        Pg, Mg = P[gt_idx], np.stack([Mraw[i] for i in gt_idx])
        ok = yr > 0
        R['R2'] = {}
        R['R2']['year_from_atlas'] = cv_acc(Pg[ok], yr[ok], 'year from atlas')
        R['R2']['year_from_marginal'] = cv_acc(Mg[ok], yr[ok], 'year from marginals')

        def drift(F):
            F = np.nan_to_num(F)
            Z = (F - F[ok].mean(0)) / (F[ok].std(0) + 1e-9)
            mus = np.stack([Z[ok][yr[ok] == y].mean(0) for y in sorted(set(yr[ok]))])
            return float(np.linalg.norm(mus[-1] - mus[0]) / np.sqrt(Z.shape[1]))
        da, dm = drift(Pg), drift(Mg)
        R['R2']['drift_atlas'] = da
        R['R2']['drift_marginal'] = dm
        log(f'    2011 to 2015 shift, per standardised dimension: '
            f'atlas {da:.3f}  marginals {dm:.3f}  '
            f'(atlas is {100*da/max(dm,1e-9):.0f}% of marginals)')
    else:
        log('    gas turbine unavailable')

    # ---- R3 fault diagnosis ---------------------------------------------
    log('\n=== R3  fault diagnosis on the hydraulic rig ===')
    if 'hydraulic' in recs:
        hy = [i for i, m in enumerate(meta) if m['system'] == 'hydraulic']
        Mh = np.stack([Mraw[i] for i in hy])
        Mhw = np.stack([Mraww[i] for i in hy])
        R['R3'] = {}
        for comp in ['cooler', 'valve', 'pump', 'acc', 'stable']:
            y = np.array([meta[i].get(comp, np.nan) for i in hy])
            if not np.isfinite(y).all() or len(set(y)) < 2:
                continue
            # condition codes arrive as floats; sklearn needs discrete labels
            y = np.round(y).astype(int)
            if len(set(y)) < 2:
                continue
            log(f'  -- {comp} --')
            R['R3'][comp] = {
                'atlas': cv_acc(P[hy], y, 'atom-quantile profile'),
                'marginal': cv_acc(Mh, y, 'per-channel marginals'),
                'atlas_warped': cv_acc(Pw[hy], y, 'atlas, recalibrated'),
                'marginal_warped': cv_acc(Mhw, y, 'marginals, recalibrated'),
            }
            lo, hi = boot_ci(P[hy], y, B=100)
            R['R3'][comp]['atlas_ci'] = [lo, hi]
    else:
        log('    hydraulic rig unavailable')

    R['counts'] = {c: int((L == c).sum()) for c in sorted(set(L))}
    R['systems'] = sorted(set(L))
    with open(os.path.join(OUT, 'results_real.json'), 'w') as f:
        json.dump(R, f, indent=2, default=float)
    np.savez_compressed(os.path.join(OUT, 'profiles_real.npz'),
                        P=P, Pw=Pw, M=M, Mw=Mw, L=L,
                        atoms=np.array(atoms.ATOM_NAMES), QS=QS)
    log(f'\nwrote results_real.json [{time.time()-t0:.0f}s]')


def _find_data(name):
    """Locate each mounted dataset by a file we know it contains."""
    markers = {
        'gasturbine': 'gt_full.csv',
        'hydraulic': 'PS1.txt',
        'transformer': 'ETTh.csv',
        'wind': 'T1.csv',
        'motor': 'measures_v2.csv',
        'compressor': 'MetroPT3(AirCompressor).csv',
        'pumprig': 'valve1',
    }
    return _find(markers[name])


if __name__ == '__main__':
    main()
