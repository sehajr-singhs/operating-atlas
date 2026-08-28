"""C-MAPSS loader. Four sub-datasets = four SYSTEM CLASSES of the same engine
family, differing in how many operating regimes and how many fault modes they
carry:

    FD001  100 units   1 regime    1 fault (HPC degradation)
    FD002  260 units   6 regimes   1 fault
    FD003  100 units   1 regime    2 faults (HPC + fan)
    FD004  249 units   6 regimes   2 faults

709 units in total. This is the only public benchmark with enough same-class
units to ask whether a class prototype exists at all, which is why the pilot
runs here before anything is spent on cloud compute.
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cmapss')
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
COLS = ['unit', 'cycle'] + [f'op{i}' for i in (1, 2, 3)] + [f's{i}' for i in range(1, 22)]
CHANNELS = [f'op{i}' for i in (1, 2, 3)] + [f's{i}' for i in range(1, 22)]


def _path(name):
    for cand in (os.path.join(ROOT, 'CMaps', name), os.path.join(ROOT, name)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(name)


def load_subset(sub, split='train'):
    df = pd.read_csv(_path(f'{split}_{sub}.txt'), sep=r'\s+', header=None,
                     names=COLS, engine='python')
    return df


def units(sub, split='train', min_cycles=60):
    """Yield (unit_id, X, meta) with X the (n_cycles, 24) channel matrix."""
    df = load_subset(sub, split)
    out = []
    for uid, d in df.groupby('unit', sort=True):
        if len(d) < min_cycles:
            continue
        X = d[CHANNELS].to_numpy(np.float64)
        out.append((int(uid), X, dict(n=len(d), sub=sub, life=int(d.cycle.max()))))
    return out
