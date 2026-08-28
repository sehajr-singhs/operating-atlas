"""Where does atlas time go? Decides the cloud budget."""
import time
import numpy as np
import atoms

rng = np.random.default_rng(0)
for (n, d) in [(200, 24), (2000, 24), (20000, 12), (20000, 32)]:
    X = np.cumsum(rng.normal(size=(n, d)), 0) + rng.normal(size=(n, d))
    U = atoms._ranks(X)
    ts = {}
    for nm, fn in [('ranks', lambda: atoms._ranks(X)),
                   ('corr', lambda: atoms._corr(U)),
                   ('eta', lambda: atoms._eta_matrix(U)),
                   ('levy', lambda: atoms._levy(U)),
                   ('jump', lambda: atoms._jump(U)),
                   ('fill', lambda: atoms._fill(U)),
                   ('beta', lambda: atoms._beta(X))]:
        t0 = time.perf_counter(); fn(); ts[nm] = time.perf_counter() - t0
    tot = sum(ts.values())
    print(f'n={n:6d} d={d:3d}  total={tot*1e3:8.1f} ms  ' +
          '  '.join(f'{k}={v*1e3:.1f}' for k, v in sorted(ts.items(), key=lambda kv: -kv[1])))
