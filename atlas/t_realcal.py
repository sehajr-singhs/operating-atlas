"""
Calibrating the dimension estimator on REAL data whose answer is known.

Every "this machine is high dimensional" verdict so far rests on the estimator
being unbiased on real, noisy, unevenly sampled telemetry, and that has only
been checked on clean synthetic shapes. A CNC tool path is the test case: the
machine follows a program, so the tool centre point traces a ONE-dimensional
curve through three-dimensional space, and it traces it again on every pass.

If the estimator returns about 1 here, the high-dimensional verdicts stand and
those machines really do fill their space. If it returns 2 or 3, it inflates on
real data and every earlier number needs revisiting.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manifold_local as ml

P = os.path.expanduser('~/probe')
fs_ = sorted(glob.glob(os.path.join(P, 'cnc', 'experiment_*.csv')))
df = pd.read_csv(fs_[0])
print(f'one CNC experiment: {len(df)} samples')

groups = {
    'tool path XYZ (should be ~1)':
        ['X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition'],
    'XYZ position + velocity (~2)':
        ['X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition',
         'X1_ActualVelocity', 'Y1_ActualVelocity', 'Z1_ActualVelocity'],
    'X axis alone: pos+vel+acc (~2)':
        ['X1_ActualPosition', 'X1_ActualVelocity', 'X1_ActualAcceleration'],
    'command vs actual X (~1, servo)':
        ['X1_ActualPosition', 'X1_CommandPosition'],
}
for name, cols in groups.items():
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        continue
    X = df[cols].to_numpy(float)
    X = X[np.isfinite(X).all(1)]
    for k in (24, 48, 96):
        if len(X) < 3 * k:
            continue
        g = ml.local_geometry(X, k=k, n_probe=400, seed=0)
        print(f'  {name:<34} k={k:3d}  dim {np.median(g["fields"]["dim"]):5.2f}'
              f'   (of {g["d"]} live)')

# pool several passes of the SAME program: the body should not grow, because
# repeating a path traces the same curve again
print('\nrepeating the same program should not add dimensions:')
cols = ['X1_ActualPosition', 'Y1_ActualPosition', 'Z1_ActualPosition']
for nexp in (1, 2, 4):
    X = pd.concat([pd.read_csv(f)[cols] for f in fs_[:nexp]]).to_numpy(float)
    X = X[np.isfinite(X).all(1)]
    g = ml.local_geometry(X, k=48, n_probe=500, seed=0)
    print(f'  {nexp} experiment(s) pooled, n={len(X):5d}   '
          f'dim {np.median(g["fields"]["dim"]):5.2f}')
