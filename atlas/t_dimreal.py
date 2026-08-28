"""How many dimensions do real machines actually occupy?

The manifold picture only has purchase if the machine's variables are
constrained. This measures the intrinsic dimension of several real bodies
against their channel count, and sweeps the neighbourhood size, because k
neighbours cannot describe a body of dimension much above log2(k).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold_local as ml

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, '_pmsm_sessions.npz'))
X = z['X'][0].astype(np.float64)
print(f'PMSM motor bench: {X.shape[0]} samples, {X.shape[1]} channels')
for k in (32, 64, 128, 256, 512):
    g = ml.local_geometry(X, k=k, n_probe=700, seed=0)
    f = g['fields']
    print(f'  k={k:4d}  dim {np.median(f["dim"]):5.2f}  '
          f'curv {np.median(f["curv"]):8.4f}  tear {np.median(f["tear"]):5.2f}'
          f'  live d={g["d"]}')
