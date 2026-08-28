"""Why does appending dead channels still change the measured dimension?"""
import numpy as np
import manifold_local as ml

rng = np.random.default_rng(0)
N = 5000
tt = np.linspace(0, 60, N)
uu = 2.0 + 1.3 * np.sin(0.7 * tt) + 0.9 * np.sin(0.11 * tt)
hh = 3.0 * np.sin(0.05 * tt)
T3 = np.stack([uu * np.cos(uu), hh, uu * np.sin(uu)], 1)
T3 = T3 + 0.01 * rng.normal(size=T3.shape)
dead = 0.001 * rng.normal(size=(N, 4))
T7 = np.concatenate([T3, dead], 1)

print('channel SNR (0 = white noise, 1 = perfectly smooth)')
print('  live 3 :', np.round(ml.channel_snr(T3), 4))
print('  with 4 dead:', np.round(ml.channel_snr(T7), 4))

for win in (5, 9, 21, 51):
    print(f'  win={win:3d} ->', np.round(ml.channel_snr(T7, win=win), 4))

print('\nweights actually applied (sqrt of snr above floor)')
for floor in (0.0, 0.02, 0.2):
    s = ml.channel_snr(T7)
    w = np.sqrt(np.maximum(s - floor, 0.0))
    print(f'  floor={floor:<5} ->', np.round(w, 4))

print('\nlocal spectrum at one probe, 3 live vs 7 columns')
for nm, X in (('3 live', T3), ('7 cols', T7)):
    g = ml.local_geometry(X, k=48, n_probe=200, seed=1, snr_weight=True)
    lam = g['spectra']
    med = np.median(lam, axis=0)
    print(f'  {nm:8s} eigenvalues (median over probes):',
          np.array2string(med, precision=6, suppress_small=False))
    print(f'           ratio to smallest:',
          np.round(med / max(med[-1], 1e-30), 1))
    print(f'           dim = {np.median(g["fields"]["dim"]):.2f}')
