"""Locate the invariance leak on real fleet telemetry."""
import numpy as np
import atoms

z = np.load('C:/Users/sehaj/kaggle_dry/out/fleet/ur5e_u6_e4_s8.npz')
X = np.concatenate([z[k] for k in z.files if k.startswith('X_0_')]).astype(np.float64)
print('X', X.shape)
rng = np.random.default_rng(11)
W = atoms.warp_channels(X, rng)

# 1. is the warp actually strictly increasing per channel?
bad = []
for j in range(X.shape[1]):
    o = np.argsort(X[:, j], kind='stable')
    dw = np.diff(W[o, j])
    if (dw < -1e-12).any():
        bad.append((j, float(dw.min())))
print('non-monotone channels:', bad[:10], 'count', len(bad))

# 2. do ranks survive?
rX, rW = atoms._ranks(X), atoms._ranks(W)
print('max |rank shift| =', np.abs(rX - rW).max())

# 3. does the load coordinate survive?
tau_cols = list(range(12, 18))
lX, lW = atoms.rank_load(X, tau_cols), atoms.rank_load(W, tau_cols)
print('max |rank_load shift| =', np.abs(lX - lW).max())

# 4. single cell
A1 = atoms.atlas_unit(X, n_cells=1, min_cell=50)
B1 = atoms.atlas_unit(W, n_cells=1, min_cell=50)
for ai, nm in enumerate(atoms.ATOM_NAMES):
    print(f'  n_cells=1  {nm:>6} max|d| = {np.nanmax(np.abs(A1[..., ai]-B1[..., ai])):.2e}')

# 5. three cells
A3 = atoms.atlas_unit(X, n_cells=3, cell_coord=lX, min_cell=50)
B3 = atoms.atlas_unit(W, n_cells=3, cell_coord=lW, min_cell=50)
for ai, nm in enumerate(atoms.ATOM_NAMES):
    print(f'  n_cells=3  {nm:>6} max|d| = {np.nanmax(np.abs(A3[..., ai]-B3[..., ai])):.2e}')
