"""
The survey: which machines have an operating body, and what predicts it.

This is the paper's central experiment. For every real system we can reach, we
measure the intrinsic dimension of its operating body against its channel count,
and we record one structural fact about how it was driven:

    closed-loop + repeated programme   a CNC running a part programme, a PLC
                                       running a plant, a lift cycling doors
    closed-loop, free duty             a turbine or transformer following demand
    open-loop / rich excitation        a testbed deliberately exploring its
                                       envelope

The prediction is that the first group has bodies and the last does not, because
a controller enforcing a setpoint is a constraint and a constraint removes a
degree of freedom. If that holds across a dozen unrelated systems it is a law
worth stating, and it tells a practitioner in one number whether any of the rest
of this applies to their data.

Dimension is measured with two published estimators, TwoNN (Facco et al. 2017)
and Levina-Bickel MLE, and we report both. Agreement is the check. An estimator
of our own devising is what produced the earlier and wrong conclusion that no
machine has a body, so nothing here rests on one.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manifold_local as ml

IN = os.environ.get('IOO_IN', '/kaggle/input')
IN_DIRS = [d.strip() for d in IN.replace(';', os.pathsep).split(os.pathsep) if d.strip()]
if not IN_DIRS:
    IN_DIRS = [IN]
MAXN = int(os.environ.get('MAXN', 12000))

# how each system is driven, recorded before any measurement is taken
DRIVE = {
    'cnc_mill': 'closed-loop, repeated programme',
    'swat_plant': 'closed-loop, repeated programme',
    'elevator': 'closed-loop, repeated programme',
    'hydraulic_rig': 'closed-loop, repeated programme',
    'battery_cycling': 'closed-loop, repeated programme',
    'gas_turbine': 'closed-loop, free duty',
    'transformer': 'closed-loop, free duty',
    'wind_farm': 'closed-loop, free duty',
    'metro_compressor': 'closed-loop, free duty',
    'pmsm_bench': 'open-loop, rich excitation',
    'pump_rig': 'open-loop, rich excitation',
    'ur5e_robot': 'open-loop, rich excitation',
    'cmapss_turbofan': 'simulated, free duty',
    'steel_plant': 'closed-loop, free duty',
}


def find(*names, root=None):
    roots = [root] if root else IN_DIRS
    for n in names:
        for r in roots:
            hits = glob.glob(os.path.join(r, '**', n), recursive=True)
            if hits:
                return sorted(hits, key=len)[0]
    return None


def numeric(df, drop=()):
    out = []
    for c in df.columns:
        if c in drop or str(c).lower().startswith('unnamed'):
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().mean() > 0.9 and s.std(skipna=True) > 1e-12:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
def L_cnc():
    fs_ = []
    for r in IN_DIRS:
        fs_ = sorted(glob.glob(os.path.join(r, '**', 'experiment_*.csv'),
                               recursive=True))
        if fs_:
            break
    if not fs_:
        return None
    d0 = pd.read_csv(fs_[0])
    cols = numeric(d0, drop=('Machining_Process',))
    return [pd.read_csv(f)[cols].to_numpy(float) for f in fs_[:8]]


def L_swat():
    """A six-stage water treatment plant under PLC control, 51 sensors and
    actuators. The published file names are plain, so an earlier glob for
    'SWaT_Dataset_*' matched nothing and the flagship closed-loop system was
    silently absent from the survey."""
    f = find('normal.csv', 'attack.csv', 'merged.csv')
    if f is None:
        return None
    df = pd.read_csv(f, low_memory=False, nrows=80000)
    # some publications ship a units row above the data
    if df.iloc[0].apply(lambda v: isinstance(v, str)).mean() > 0.5:
        df = pd.read_csv(f, low_memory=False, nrows=80000, skiprows=[1])
    cols = numeric(df)
    if len(cols) < 5:
        return None
    V = df[cols].to_numpy(float)
    return [V[i:i + 12000] for i in range(0, min(len(V), 60000), 12000)]


def L_elevator():
    f = find('predictive-maintenance-dataset.csv', '*elevator*.csv')
    if f is None:
        return None
    df = pd.read_csv(f, low_memory=False)
    cols = numeric(df)
    V = df[cols].to_numpy(float)
    return [V[i:i + 12000] for i in range(0, min(len(V), 60000), 12000)]


def L_hydraulic():
    spec = {'PS1': 100, 'PS2': 100, 'PS3': 100, 'EPS1': 100, 'FS1': 10,
            'FS2': 10, 'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1, 'VS1': 1,
            'CE': 1, 'CP': 1, 'SE': 1}
    mats = []
    for nm, rate in spec.items():
        f = find(f'{nm}.txt')
        if f is None:
            continue
        A = np.loadtxt(f)
        k = rate // 10 if rate > 10 else 1
        if k > 1:
            A = A[:, :(A.shape[1] // k) * k].reshape(len(A), -1, k).mean(2)
        elif rate < 10:
            A = np.repeat(A, 10 // rate, axis=1)
        mats.append(A.astype(np.float32))
    if len(mats) < 5:
        return None
    m = min(a.shape[1] for a in mats); n = min(len(a) for a in mats)
    C = np.stack([a[:n, :m] for a in mats], -1)
    return [C[i:i + 30].reshape(-1, C.shape[-1]).astype(float)
            for i in range(0, min(n, 300), 30)]


def L_battery():
    f = find('*Battery Cycle Life.csv', '*battery*.csv')
    if f is None:
        return None
    df = pd.read_csv(f)
    idc = [c for c in df.columns if 'battery' in c.lower()]
    cols = [c for c in numeric(df) if c not in ('cycle', 'cycle_life')]
    if not idc:
        return [df[cols].to_numpy(float)]
    out = []
    for _, g in list(df.groupby(idc[0]))[:6]:
        if len(g) > 500:
            out.append(g[cols].to_numpy(float))
    return out or None


def L_gt():
    f = find('gt_full.csv')
    if f is None:
        return None
    df = pd.read_csv(f)
    V = df[numeric(df)].to_numpy(float)
    return [V[i:i + 6000] for i in range(0, min(len(V), 30000), 6000)]


def L_ett():
    f = find('ETTh.csv', 'ETTh1.csv')
    if f is None:
        return None
    df = pd.read_csv(f)
    V = df[numeric(df, drop=('date',))].to_numpy(float)
    return [V[i:i + 5000] for i in range(0, min(len(V), 15000), 5000)]


def L_wind():
    f = find('T1.csv')
    if f is None:
        return None
    df = pd.read_csv(f)
    V = df[numeric(df, drop=('Date/Time',))].to_numpy(float)
    return [V[i:i + 8000] for i in range(0, min(len(V), 40000), 8000)]


def L_metro():
    f = find('MetroPT3(AirCompressor).csv', 'MetroPT3.csv')
    if f is None:
        return None
    df = pd.read_csv(f, nrows=200000)
    V = df[numeric(df, drop=('timestamp',))].to_numpy(float)
    return [V[i:i + 12000] for i in range(0, min(len(V), 60000), 12000)]


def L_pmsm():
    f = find('measures_v2.csv')
    if f is None:
        return None
    df = pd.read_csv(f)
    cols = [c for c in numeric(df) if c != 'profile_id']
    out = []
    for _, g in list(df.groupby('profile_id'))[:6]:
        if len(g) > 8000:
            out.append(g[cols].to_numpy(float)[:12000])
    return out or None


def L_skab():
    fs_ = []
    for r in IN_DIRS:
        fs_ = sorted(glob.glob(os.path.join(r, '**', 'valve1', '*.csv'),
                               recursive=True))[:8]
        if fs_:
            break
    if not fs_:
        return None
    out = []
    for f in fs_:
        d = pd.read_csv(f, sep=';')
        cols = [c for c in d.columns
                if c not in ('datetime', 'anomaly', 'changepoint')]
        out.append(d[cols].to_numpy(float))
    return out


def L_cmapss():
    f = find('train_FD002.txt', 'train_FD001.txt')
    if f is None:
        return None
    names = ['unit', 'cycle'] + [f'op{i}' for i in (1, 2, 3)] + \
            [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(f, sep=r'\s+', header=None, names=names, engine='python')
    ch = names[2:]
    out = []
    for _, g in list(df.groupby('unit'))[:40]:
        if len(g) > 150:
            out.append(g[ch].to_numpy(float))
    # C-MAPSS units are short, so several are pooled to make one body
    return [np.concatenate(out[i:i + 10]) for i in range(0, len(out), 10)] or None


def L_steel():
    f = find('Steel_industry_data.csv', '*steel*.csv')
    if f is None:
        return None
    df = pd.read_csv(f)
    V = df[numeric(df)].to_numpy(float)
    return [V[i:i + 8000] for i in range(0, min(len(V), 32000), 8000)]


LOADERS = {
    'cnc_mill': L_cnc, 'swat_plant': L_swat, 'elevator': L_elevator,
    'hydraulic_rig': L_hydraulic, 'battery_cycling': L_battery,
    'gas_turbine': L_gt, 'transformer': L_ett, 'wind_farm': L_wind,
    'metro_compressor': L_metro, 'pmsm_bench': L_pmsm, 'pump_rig': L_skab,
    'cmapss_turbofan': L_cmapss, 'steel_plant': L_steel,
}


def embed(X):
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(1)]
    if len(X) > MAXN:
        X = X[::max(1, len(X) // MAXN)][:MAXN]
    if len(X) < 600 or X.shape[1] < 3:
        return None, 0, X.shape[1] if X.ndim == 2 else 0
    w = np.sqrt(np.maximum(ml.channel_snr(X) - 0.02, 0.0))
    live = w > 1e-6
    if live.sum() < 3:
        live = np.ones(X.shape[1], bool); w = np.ones(X.shape[1])
    return ml._ranks(X)[:, live] * w[live], int(live.sum()), X.shape[1]


def dims(Z):
    from sklearn.neighbors import NearestNeighbors
    two = ml.intrinsic_dim_twonn(Z)
    mles = []
    for k in (16, 32, 64):
        if len(Z) < 4 * k:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        d, _ = nn.kneighbors(Z[::max(1, len(Z) // 1200)])
        v = [ml.local_dim_mle(r[1:]) for r in d]
        mles.append(np.nanmedian(v))
    return two, (float(np.nanmedian(mles)) if mles else float('nan'))


def main():
    rows = []
    print(f'{"system":<20}{"drive":<34}{"chan":>5}{"live":>5}'
          f'{"TwoNN":>7}{"MLE":>7}{"ratio":>7}  verdict')
    for name, fn in LOADERS.items():
        try:
            recs = fn()
        except Exception as e:
            print(f'{name:<20}load failed: {type(e).__name__}: {e}')
            continue
        if not recs:
            print(f'{name:<20}not found')
            continue
        per = []
        for X in recs[:6]:
            Z, dlive, dall = embed(X)
            if Z is None or dlive < 3:
                continue
            t, m = dims(Z)
            if np.isfinite(t):
                per.append((t, m, dlive, dall))
        if not per:
            print(f'{name:<20}no usable records')
            continue
        t = float(np.median([p[0] for p in per]))
        m = float(np.nanmedian([p[1] for p in per]))
        dlive = int(np.median([p[2] for p in per]))
        dall = int(np.median([p[3] for p in per]))
        ratio = t / max(dlive, 1)
        verdict = ('BODY' if ratio < 0.35 else
                   'thin' if ratio < 0.6 else 'CLOUD')
        drive = DRIVE.get(name, '?')
        print(f'{name:<20}{drive:<34}{dall:>5}{dlive:>5}{t:>7.2f}{m:>7.2f}'
              f'{ratio:>7.2f}  {verdict}')
        rows.append(dict(system=name, drive=drive, channels=dall, live=dlive,
                         twonn=t, mle=m, ratio=ratio, verdict=verdict,
                         n_records=len(per)))

    if rows:
        import json
        out = os.environ.get('IOO_OUT', '/kaggle/working')
        out = os.environ.get('IOO_OUT', os.getcwd())
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'survey.json'), 'w') as f:
            json.dump(rows, f, indent=2)
        print('\nby drive regime:')
        for g in ['closed-loop, repeated programme', 'closed-loop, free duty',
                  'open-loop, rich excitation', 'simulated, free duty']:
            rs = [r['ratio'] for r in rows if r['drive'] == g]
            if rs:
                print(f'  {g:<34} n={len(rs)}  median ratio '
                      f'{np.median(rs):.2f}   [{min(rs):.2f}, {max(rs):.2f}]')


if __name__ == '__main__':
    main()
