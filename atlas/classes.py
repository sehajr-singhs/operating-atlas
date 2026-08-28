"""
The index of operators: class signatures that compare across systems with
different numbers of channels.

A unit's atlas is a vector over its own channel pairs, so two systems with
different channel counts have atlases of different length and cannot be
compared directly. That is a real obstacle to any class-level claim, because a
motor has 12 channels, a pump testbed has 8 and a turbofan has 24.

The fix is to stop treating the atlas as a vector over pairs and start treating
it as a DISTRIBUTION over pairs. For each atom, take a fixed set of quantiles of
its values across all pairs. The result is 9 atoms x 9 quantiles = 81 numbers
whatever the channel count, and it says what kind of relational geometry the
machine has rather than which specific pair does what: how much of this
machine's behaviour is single valued, how much circulates, how spread its
timescales are.

That is exactly the class-level object. A specific unit is then this profile
plus its own per-pair detail, which is the 'base class fine-tunes to the unit'
structure stated in a way that survives a change of instrumentation entirely.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import atoms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
QS = np.array([0.02, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.98])
NA = len(atoms.ATOM_NAMES)


def profile(X, n_cells=1, min_len=400, fixed_n=None):
    """Channel-count-free class signature of one record.

    `fixed_n` strides every record down to the same number of samples. Without
    it the comparison across systems is confounded: `fill` counts occupied grid
    cells and so grows with record length, and `tau` is an autocorrelation time
    measured in samples and so tracks the sampling rate. A classifier could then
    separate a motor bench from a turbofan by how fast each was logged rather
    than by any physics, and would look excellent doing it.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(1)]
    if len(X) < min_len or X.shape[1] < 3:
        return None
    if fixed_n is not None:
        if len(X) < fixed_n:
            return None
        X = X[::max(1, len(X) // fixed_n)][:fixed_n]
    keep = X.std(0) > 1e-10
    X = X[:, keep]
    if X.shape[1] < 3:
        return None
    A = atoms.atlas_unit(X, n_cells=n_cells, min_cell=100)[0]
    out = np.empty((NA, len(QS)))
    for ai in range(NA):
        v = A[:, ai]
        v = v[np.isfinite(v)]
        out[ai] = np.quantile(v, QS) if len(v) >= 5 else np.nan
    return out.reshape(-1)


# ---------------------------------------------------------------------------
def load_pmsm():
    z = np.load(os.path.join(HERE, '_pmsm_sessions.npz'))
    return [('pmsm', z['X'][i].astype(np.float64)) for i in range(len(z['X']))]


def load_skab():
    out = []
    for sub in ['valve1', 'valve2', 'other']:
        for f in sorted(glob.glob(os.path.join(
                DATA, 'skoltech-anomaly-benchmark-skab', 'SKAB', sub, '*.csv'))):
            d = pd.read_csv(f, sep=';')
            cols = [c for c in d.columns
                    if c not in ('datetime', 'anomaly', 'changepoint')]
            out.append((f'skab-{sub}', d[cols].to_numpy(np.float64)))
    return out


def load_metropt(win=20000, stride=20000, max_win=40):
    f = os.path.join(DATA, 'metropt-3-dataset', 'MetroPT3(AirCompressor).csv')
    d = pd.read_csv(f)
    cols = [c for c in d.columns if c not in ('Unnamed: 0', 'timestamp')
            and d[c].dtype != object]
    V = d[cols].to_numpy(np.float64)
    return [('metropt', V[i:i + win])
            for i in range(0, min(len(V) - win, stride * max_win), stride)]


def load_cmapss(max_units=60):
    import data_cmapss as dc
    out = []
    for sub in dc.SUBSETS:
        for uid, X, meta in dc.units(sub, min_cycles=150)[:max_units]:
            out.append((f'cmapss-{sub}', X))
    return out


def load_finance(win=500, stride=110):
    """Included to test whether the vocabulary is machine specific. Channels are
    the six assets' log returns and realised ranges, so a 'unit' is a window of
    the market rather than a piece of equipment."""
    cache = os.path.expanduser('~/Documents/algo-trading-crossasset/cache')
    fs_ = sorted(glob.glob(os.path.join(cache, '*.csv')))
    if not fs_:
        return []
    frames = []
    for f in fs_:
        d = pd.read_csv(f, parse_dates=['Date']).set_index('Date')
        nm = os.path.basename(f).replace('.csv', '')
        lr = np.log(d['Close']).diff()
        rng_ = (np.log(d['High']) - np.log(d['Low']))
        frames.append(pd.DataFrame({f'{nm}_r': lr, f'{nm}_v': rng_}))
    M = pd.concat(frames, axis=1).dropna()
    V = M.to_numpy(np.float64)
    return [('finance', V[i:i + win])
            for i in range(0, max(0, len(V) - win), stride)]


LOADERS = [('pmsm', load_pmsm), ('skab', load_skab), ('metropt', load_metropt),
           ('cmapss', load_cmapss), ('finance', load_finance)]


if __name__ == '__main__':
    import time
    import json
    t0 = time.time()
    FIXED_N = int(os.environ.get('FIXED_N', 900))
    P, L = [], []
    for nm, fn in LOADERS:
        try:
            recs = fn()
        except Exception as e:
            print(f'  {nm}: SKIPPED ({e})', flush=True); continue
        k = 0
        for lab, X in recs:
            p = profile(X, fixed_n=FIXED_N)
            if p is not None and np.isfinite(p).all():
                P.append(p); L.append(lab); k += 1
        print(f'  {nm}: {k} records, {len(recs)} offered  [{time.time()-t0:.0f}s]',
              flush=True)
    P = np.array(P); L = np.array(L)
    print(f'\nprofiles {P.shape}, {len(set(L))} classes')
    for c in sorted(set(L)):
        print(f'  {c:<16} n={int((L == c).sum())}')

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    keep = np.array([int((L == c).sum()) >= 5 for c in L])
    Pk, Lk = P[keep], L[keep]
    nf = int(min(5, pd.Series(Lk).value_counts().min()))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    cv = StratifiedKFold(nf, shuffle=True, random_state=0)
    s = cross_val_score(clf, Pk, Lk, cv=cv)
    maj = pd.Series(Lk).value_counts().max() / len(Lk)
    print(f'\nclass accuracy = {s.mean()*100:.1f} +- {s.std()*100:.1f} % '
          f'({nf}-fold, majority {maj*100:.1f} %, {len(set(Lk))} classes, '
          f'records equalised to {FIXED_N} samples)')

    # ---- ablation: is this physics or is it the logger? ------------------
    # tau is an autocorrelation time in samples and fill grows with record
    # length, so both can encode acquisition settings rather than mechanism.
    print('\nleave-one-atom-out and one-atom-only:')
    Q = len(QS)
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        cols = np.arange(Pk.shape[1])
        blk = (cols // Q) == ai
        a_only = cross_val_score(clf, Pk[:, blk], Lk, cv=cv).mean()
        a_drop = cross_val_score(clf, Pk[:, ~blk], Lk, cv=cv).mean()
        print(f'  {nm:>6}  only = {a_only*100:5.1f} %   without = {a_drop*100:5.1f} %')
    ablation = {}
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        cols = np.arange(Pk.shape[1])
        blk = (cols // Q) == ai
        ablation[nm] = dict(
            only=float(cross_val_score(clf, Pk[:, blk], Lk, cv=cv).mean()),
            without=float(cross_val_score(clf, Pk[:, ~blk], Lk, cv=cv).mean()))
    rate_free = [i for i, nm in enumerate(atoms.ATOM_NAMES)
                 if nm not in ('tau', 'fill')]
    cols = np.arange(Pk.shape[1])
    m = np.isin(cols // Q, rate_free)
    a_rf = cross_val_score(clf, Pk[:, m], Lk, cv=cv)
    print(f'  without tau AND fill (the two acquisition-sensitive atoms): '
          f'{a_rf.mean()*100:.1f} +- {a_rf.std()*100:.1f} %')

    # confusion, to see whether the hard classes (same rig, different fault)
    # are actually being told apart or are being carried by the easy ones
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_predict
    pred = cross_val_predict(clf, Pk, Lk, cv=cv)
    labs = sorted(set(Lk))
    cm = confusion_matrix(Lk, pred, labels=labs)
    print('\nconfusion (rows true):')
    print('        ' + ' '.join(f'{c[:7]:>8}' for c in labs))
    for r, c in zip(cm, labs):
        print(f'  {c[:7]:>7} ' + ' '.join(f'{x:>8d}' for x in r))
    np.savez(os.path.join(HERE, 'class_profiles.npz'), P=P, L=L, QS=QS,
             atoms=np.array(atoms.ATOM_NAMES))
    json.dump(dict(acc=float(s.mean()), sd=float(s.std()), majority=float(maj),
                   n_classes=int(len(set(Lk))), n=int(len(Lk)),
                   fixed_n=FIXED_N, ablation=ablation,
                   acc_no_tau_fill=float(a_rf.mean()),
                   sd_no_tau_fill=float(a_rf.std()),
                   confusion=cm.tolist(), confusion_labels=list(labs),
                   counts={c: int((Lk == c).sum()) for c in labs}),
              open(os.path.join(HERE, 'class_results.json'), 'w'), indent=2)
    print('saved class_profiles.npz')
