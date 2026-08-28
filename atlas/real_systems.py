"""
Loaders for real industrial telemetry, one function per physical system.

Every loader returns a list of (label, X, meta) with X an (n, d) float array of
raw channels. Loaders are written defensively: numeric columns are detected
rather than assumed, constant and identifier columns are dropped, and a system
that fails to load is skipped with a message instead of taking the run down.

What each system contributes, and why it is here:

  gasturbine  a real industrial gas turbine logged hourly for five years, with
              compressor discharge pressure, turbine inlet and exhaust
              temperature, and ambient conditions. This is the only genuinely
              REAL turbomachinery in the study, and the five-year span is real
              asset and sensor drift rather than a simulated warp.
  hydraulic   a real hydraulic test rig, 17 sensors at three different sampling
              rates spanning pressure, flow, temperature, vibration and motor
              power, with documented ground-truth fault conditions on four
              components. The richest multi-physics real system available.
  pmu         real synchrophasor measurements from a power-grid testbed.
  ett         a real electricity transformer, six load channels and oil
              temperature.
  wind        a real wind farm SCADA feed. Only four numeric channels, so it
              contributes the fewest pairs of any system and is reported as
              such rather than quietly padded.
"""

import glob
import os
import numpy as np
import pandas as pd

# UCI publishes the gas turbine as one file per year; the mirrored copy is the
# five concatenated in order, so the year boundaries are the published counts.
GT_YEAR_COUNTS = [7411, 7628, 7152, 7158, 7384]
GT_YEARS = [2011, 2012, 2013, 2014, 2015]


def _numeric(df, drop=()):
    out = []
    for c in df.columns:
        if c in drop or str(c).lower() in ('', 'unnamed: 0', 'index', 'id'):
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().mean() > 0.9 and s.std(skipna=True) > 1e-12:
            out.append(c)
    return out


def _find(root, *names):
    for n in names:
        hits = glob.glob(os.path.join(root, '**', n), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    return None


# ---------------------------------------------------------------------------
def load_gasturbine(root, seg=1100):
    f = _find(root, 'gt_full.csv', 'gt_*.csv')
    if f is None:
        raise FileNotFoundError('gas turbine csv')
    df = pd.read_csv(f)
    cols = _numeric(df)
    V = df[cols].to_numpy(np.float64)
    # attach the year each row belongs to, for the real-drift experiment
    year = np.concatenate([np.full(c, y) for c, y in zip(GT_YEAR_COUNTS, GT_YEARS)])
    if len(year) != len(V):
        year = np.full(len(V), -1)
    out = []
    for s in range(0, len(V) - seg, seg):
        out.append(('gasturbine', V[s:s + seg],
                    dict(year=int(np.median(year[s:s + seg])), channels=cols)))
    return out


def load_hydraulic(root, cycles_per_record=20, hz=10):
    """The rig logs each 60 s cycle as one ROW per sensor file, with the columns
    being samples within that cycle. Sensors run at 100, 10 and 1 Hz, so every
    channel is brought to a common 10 Hz base before any pair is formed: the
    fast ones by block mean, the slow ones by repetition."""
    spec = {'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100,
            'PS6': 100, 'EPS1': 100, 'FS1': 10, 'FS2': 10,
            'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1, 'VS1': 1,
            'CE': 1, 'CP': 1, 'SE': 1}
    mats, names = [], []
    for nm, rate in spec.items():
        f = _find(root, f'{nm}.txt')
        if f is None:
            continue
        A = np.loadtxt(f)
        if rate > hz:
            k = rate // hz
            A = A[:, :(A.shape[1] // k) * k].reshape(len(A), -1, k).mean(2)
        elif rate < hz:
            A = np.repeat(A, hz // rate, axis=1)
        mats.append(A.astype(np.float32)); names.append(nm)
    if len(mats) < 5:
        raise FileNotFoundError('hydraulic sensors')
    m = min(a.shape[1] for a in mats)
    n_cyc = min(len(a) for a in mats)
    C = np.stack([a[:n_cyc, :m] for a in mats], -1)     # (cycles, samples, d)

    prof = _find(root, 'profile.txt')
    P = np.loadtxt(prof)[:n_cyc] if prof else np.zeros((n_cyc, 5))

    out = []
    for s in range(0, n_cyc - cycles_per_record, cycles_per_record):
        blk = C[s:s + cycles_per_record]
        X = blk.reshape(-1, blk.shape[-1]).astype(np.float64)
        lab = P[s:s + cycles_per_record]
        # These are categorical condition codes, so the block summary must be
        # the MODE. A median over a block that straddles a condition change
        # returns a value halfway between two codes, which is not a class at
        # all and makes the label continuous.
        def mode(col):
            v, c = np.unique(lab[:, col], return_counts=True)
            return float(v[np.argmax(c)])
        out.append(('hydraulic', X, dict(
            channels=names, cooler=mode(0), valve=mode(1),
            pump=mode(2), acc=mode(3), stable=mode(4))))
    return out


def load_pmu(root, seg=1500):
    """The raw synchrophasor capture stores most of its payload as strings, so
    picking the first file by name yielded two usable channels out of a
    24-channel measurement. Every candidate file is scanned and the one with the
    most genuinely numeric, non-constant columns wins."""
    best, best_cols, best_df = None, [], None
    for cand in ('Clean_Raw.csv', 'Clean_FDI_TSA_Combined.csv',
                 'Clean_TSA_Combined.csv', 'Clean_FDI_Combined.csv'):
        f = _find(root, cand)
        if f is None:
            continue
        try:
            df = pd.read_csv(f, low_memory=False, nrows=20000)
        except Exception:
            continue
        cols = _numeric(df)
        if len(cols) > len(best_cols):
            best, best_cols, best_df = f, cols, df
    if best is None or len(best_cols) < 3:
        raise FileNotFoundError('pmu csv with usable numeric channels')
    V = best_df[best_cols[:24]].to_numpy(np.float64)
    return [('pmu', V[s:s + seg], dict(channels=best_cols[:24]))
            for s in range(0, len(V) - seg, seg)]


def load_ett(root, seg=1200):
    f = _find(root, 'ETTh.csv', 'ETTh1.csv')
    if f is None:
        raise FileNotFoundError('ETT csv')
    df = pd.read_csv(f)
    cols = _numeric(df, drop=('date',))
    V = df[cols].to_numpy(np.float64)
    return [('transformer', V[s:s + seg], dict(channels=cols))
            for s in range(0, len(V) - seg, seg)]


def load_wind(root, seg=1200):
    f = _find(root, 'T1.csv')
    if f is None:
        raise FileNotFoundError('wind csv')
    df = pd.read_csv(f)
    cols = _numeric(df, drop=('Date/Time',))
    V = df[cols].to_numpy(np.float64)
    return [('wind', V[s:s + seg], dict(channels=cols))
            for s in range(0, len(V) - seg, seg)]


def load_pmsm(root, seg=4000, max_sessions=40):
    """The public bench is ONE 52 kW motor recorded in 69 measurement sessions,
    so a session is an operating episode and not a separate machine."""
    f = _find(root, 'measures_v2.csv')
    if f is None:
        raise FileNotFoundError('pmsm csv')
    df = pd.read_csv(f)
    cols = [c for c in _numeric(df) if c != 'profile_id']
    out = []
    for pid, d in list(df.groupby('profile_id', sort=True)):
        if len(d) < seg:
            continue
        V = d[cols].to_numpy(np.float64)[:seg]
        out.append(('motor', V, dict(channels=cols, session=int(pid))))
        if len(out) >= max_sessions:
            break
    return out


def load_metropt(root, seg=6000, max_rec=40):
    f = _find(root, 'MetroPT3(AirCompressor).csv', 'MetroPT3.csv', 'metropt3.csv')
    if f is None:
        raise FileNotFoundError('metropt csv')
    df = pd.read_csv(f)
    cols = _numeric(df, drop=('timestamp', 'Unnamed: 0'))
    V = df[cols].to_numpy(np.float64)
    step = max(seg, len(V) // max_rec)
    return [('compressor', V[s:s + seg], dict(channels=cols))
            for s in range(0, len(V) - seg, step)][:max_rec]


def load_skab(root, seg=1000):
    """The published SKAB CSVs carry NaN rows (missing measurements); rows with
    any missing value are dropped before the length check, otherwise the common
    finite-length stride below discards every record."""
    out = []
    for sub in ('valve1', 'valve2', 'other'):
        for f in sorted(glob.glob(os.path.join(root, '**', sub, '*.csv'),
                                  recursive=True)):
            d = pd.read_csv(f, sep=';')
            cols = [c for c in d.columns
                    if c not in ('datetime', 'anomaly', 'changepoint')]
            V = d[cols].to_numpy(np.float64)
            V = V[np.isfinite(V).all(1)]
            if len(V) >= seg:
                out.append(('pumprig', V[:seg], dict(channels=cols, sub=sub)))
    if not out:
        raise FileNotFoundError('skab csvs')
    return out


def load_cnc(root, seg=1000, max_rec=40):
    """CNC milling tool-wear dataset (Goecks et al. 2019). Each run is one
    cutting operation with sensor channels for position, spindle speed,
    feed rate and vibration."""
    f = _find(root, 'analyte1.csv', 'experiment_*.csv', 'Sensor*.csv')
    if f is None:
        # try a directory of per-run CSVs
        csvs = sorted(glob.glob(os.path.join(root, '**', '*.csv'),
                                recursive=True))
        if not csvs:
            raise FileNotFoundError('cnc milling csv')
        out = []
        for fp in csvs:
            try:
                d = pd.read_csv(fp)
            except Exception:
                continue
            cols = _numeric(d)
            if len(cols) < 3:
                continue
            V = d[cols].to_numpy(np.float64)
            V = V[np.isfinite(V).all(1)]
            if len(V) >= seg:
                out.append(('cnc_mill', V[:seg],
                            dict(channels=cols, run=os.path.basename(fp))))
            if len(out) >= max_rec:
                break
        if not out:
            raise FileNotFoundError('cnc milling csv with usable channels')
        return out
    d = pd.read_csv(f)
    cols = _numeric(d)
    V = d[cols].to_numpy(np.float64)
    V = V[np.isfinite(V).all(1)]
    step = max(seg, len(V) // max_rec)
    return [('cnc_mill', V[s:s + seg], dict(channels=cols))
            for s in range(0, len(V) - seg, step)][:max_rec]


REAL_LOADERS = {
    'gasturbine': load_gasturbine,
    'hydraulic': load_hydraulic,
    'transformer': load_ett,
    'wind': load_wind,
    'motor': load_pmsm,
    'compressor': load_metropt,
    'pumprig': load_skab,
    'cnc_mill': load_cnc,
}
