"""Generate the robot datasets. The main guard is required: on Windows the
process pool spawns fresh interpreters that re-import this module, and without
it each child re-runs the generation loop and the pool tears itself down."""
import os
import time
import numpy as np
from robot_sim import build_dataset

OUT = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    for plat in ['ur5e', 'panda', 'iiwa14']:
        t0 = time.time()
        X, Y, S, G, L = build_dataset(plat, n_ep=48, seconds=180.0, workers=6)
        np.savez_compressed(os.path.join(OUT, f'robot_{plat}.npz'),
                            X=X, Y=Y, S=S, G=G, L=L)
        print(f'{plat}: X{X.shape} Y{Y.shape} S{S.shape} '
              f'regimes {np.bincount(L, minlength=5)} ({time.time()-t0:.0f}s)',
              flush=True)


if __name__ == '__main__':
    main()
