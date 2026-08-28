import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import mujoco
import robot_sim as rs

for plat in ['ur5e', 'panda', 'iiwa14']:
    m, nj, cfg = rs._load(plat)
    d = mujoco.MjData(m)
    print(f'{plat}: dt={m.opt.timestep}  nq={m.nq} nv={m.nv} nbody={m.nbody} '
          f'ngeom={m.ngeom} solver={m.opt.solver} iterations={m.opt.iterations}')
    N = 3000
    t0 = time.perf_counter()
    for _ in range(N):
        mujoco.mj_step(m, d)
    t_bare = time.perf_counter() - t0
    d2 = mujoco.MjData(m)
    t0 = time.perf_counter()
    mujoco.mj_step(m, d2, nstep=N)
    t_batch = time.perf_counter() - t0
    print(f'   bare python loop : {N/t_bare:8.0f} steps/s')
    print(f'   nstep=N in C     : {N/t_batch:8.0f} steps/s  '
          f'({t_bare/t_batch:.1f}x)')
