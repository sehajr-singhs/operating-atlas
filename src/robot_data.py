"""
A controlled multi-physics robot testbed.

A planar two-link arm, hinged about the vertical axis so gravity is neutral and
the dynamics are set by actuation, contact and heat rather than by a constant
load. Three physical processes are coupled:

  mechanical  rigid-body dynamics, integrated by MuJoCo
  contact     a wall the tip strikes when extended; stiff, frictional, and the
              source of the sharpest change in the local geometry
  thermal     per-motor winding temperature driven by ohmic heating and
              Newtonian cooling,
                  dT_i/dt = alpha * tau_i^2 - beta * (T_i - T_env),
              with a time constant of seconds against a control bandwidth of
              milliseconds -- the stiff multi-scale coupling that makes this a
              multi-physics problem rather than two separate ones.

Actuator noise is state dependent: PWM and brush noise grow with speed, and a
motor that is hot is derated, so the noise is neither additive nor isotropic.
That is what gives V(s) genuine structure for the diffusion metric to see.

Ground-truth regime labels are recorded but never shown to any model. They
exist only to ask, afterwards, whether the invariants found the physics.
"""

import numpy as np

XML = """
<mujoco>
  <option timestep="0.002" integrator="RK4"/>
  <default><geom friction="0.9 0.02 0.001" solref="0.02 1"/></default>
  <worldbody>
    <geom name="wall" type="box" size="0.02 0.5 0.3" pos="0.50 0 0" rgba=".7 .3 .3 1"/>
    <body name="l1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.15"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.35 0 0" size="0.04" mass="1"/>
      <body name="l2" pos="0.35 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" damping="0.10" range="-2.6 2.6"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.30 0 0" size="0.035" mass="0.7"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="12" ctrlrange="-1 1"/>
    <motor joint="j2" gear="9"  ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

ALPHA, BETA = 0.20, 0.05           # ohmic gain (per N^2 m^2 s), cooling rate (1/s)
# Equilibrium rise is ALPHA/BETA * <tau^2> = 4 <tau^2>. Across episodes the
# commanded amplitude varies enough that <tau^2> spans roughly 4 to 30, so
# steady-state winding temperature spans ~40 C to ~140 C and the derating
# knee is genuinely crossed on high-duty episodes rather than never.
T_ENV_BASE = 22.0
DERATE_T = 95.0                    # torque derating knee


def rollout(n_steps, seed=0, decim=10, t_env=None):
    """One episode. Returns the state matrix, the regime labels and a dict of
    per-channel names. Sampling is decimated to 50 Hz, which is the rate a real
    drive would report telemetry at."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    rng = np.random.RandomState(seed)
    dt = m.opt.timestep
    # Start the arm swung away from the wall. The tip reaches x = 0.65 when
    # fully extended and the wall's inner face is at x = 0.48, so an unlucky
    # initial pose starts *inside* the wall and the episode is pinned by a
    # penetration force from the first step.
    d.qpos[:] = [0.9 + 0.25 * rng.randn(), 0.5 + 0.25 * rng.randn()]
    mujoco.mj_forward(m, d)
    if d.ncon:
        raise RuntimeError(f'seed {seed} starts in contact (ncon={d.ncon})')
    T = np.array([T_ENV_BASE, T_ENV_BASE]) + rng.randn(2) * 1.5
    t_env = T_ENV_BASE + (rng.rand() * 8 - 3 if t_env is None else t_env)

    # a smooth random command signal: sum of a few sinusoids per joint
    nf = 4
    freq = rng.uniform(0.15, 1.4, (2, nf))
    phase = rng.uniform(0, 2 * np.pi, (2, nf))
    amp = rng.uniform(0.2, 0.75, (2, nf))
    amp /= amp.sum(1, keepdims=True) / rng.uniform(0.55, 1.0, (2, 1))

    rows, labs = [], []
    for i in range(n_steps):
        t = i * dt
        u = (amp * np.sin(2 * np.pi * freq * t + phase)).sum(1)

        # thermal derating: a hot motor cannot deliver full torque
        derate = 1.0 / (1.0 + np.exp((T - DERATE_T) / 6.0))
        u = np.clip(u, -1, 1) * derate

        # state-dependent actuator noise: grows with speed, and with heat
        spd = np.abs(d.qvel)
        sd = 0.012 + 0.020 * spd / (1.0 + spd) + 0.010 * np.clip((T - 60) / 40, 0, 1)
        u_noisy = np.clip(u + rng.randn(2) * sd, -1, 1)
        d.ctrl[:] = u_noisy
        mujoco.mj_step(m, d)

        # qfrc_actuator is the generalised joint torque, which is what the
        # winding current is proportional to; actuator_force is the pre-gear
        # control effort and would mis-scale the ohmic term by gear^2.
        tau = d.qfrc_actuator.copy()
        T = T + dt * (ALPHA * tau ** 2 - BETA * (T - t_env))

        if i % decim == 0:
            ncon = d.ncon
            cf = 0.0
            if ncon:
                import mujoco as mj
                buf = np.zeros(6)
                for c in range(ncon):
                    mj.mj_contactForce(m, d, c, buf)
                    cf += abs(buf[0])
            rows.append(np.concatenate([d.qpos.copy(), d.qvel.copy(), tau,
                                        T.copy(), [t_env, cf]]))
            # ground-truth regime, recorded but never used as a model input
            v = np.abs(d.qvel).max()
            if cf > 1e-6:
                lab = 2                      # contact
            elif T.max() > DERATE_T - 12:
                lab = 3                      # thermally derated
            elif v > 1.6:
                lab = 1                      # fast free motion
            else:
                lab = 0                      # slow free motion
            labs.append(lab)
    names = ['q1', 'q2', 'dq1', 'dq2', 'tau1', 'tau2', 'T1', 'T2', 'Tenv', 'cforce']
    return np.array(rows), np.array(labs), names


STATE_COLS = ['q1', 'q2', 'dq1', 'dq2', 'tau1', 'tau2', 'T1', 'T2', 'Tenv']
TARGET_COLS = ['dq1_next', 'dq2_next', 'T1_next', 'T2_next']


def build_dataset(n_ep=60, n_steps=30000, seed0=0, horizon=5, decim=10):
    """Episodes -> (features, targets, state, group, regime).

    The task is a short-horizon forward model: given the operating state and a
    causal summary of its recent history, predict the joint velocities and
    winding temperatures `horizon` telemetry steps ahead. Contact force is an
    input, never a target, and the regime label is neither.
    """
    X, Y, S, G, L = [], [], [], [], []
    for e in range(n_ep):
        A, lab, names = rollout(n_steps, seed=seed0 + e, decim=decim)
        idx = {n: i for i, n in enumerate(names)}
        st = A[:, [idx[c] for c in STATE_COLS]]
        cf = A[:, [idx['cforce']]]
        # causal history features: exponential moving averages at three scales
        feats = [st, cf]
        for span in (10, 50, 200):
            a = 2.0 / (span + 1)
            ew = np.empty_like(st)
            acc = st[0].copy()
            for i in range(len(st)):
                acc = a * st[i] + (1 - a) * acc
                ew[i] = acc
            feats.append(ew)
            feats.append(st - ew)
        F = np.concatenate(feats, 1)
        tgt = np.concatenate([st[:, [2, 3]], st[:, [6, 7]]], 1)
        n = len(st) - horizon
        X.append(F[:n]); Y.append(tgt[horizon:horizon + n])
        S.append(st[:n]); L.append(lab[:n])
        G.append(np.full(n, e))
    return (np.concatenate(X).astype(np.float32),
            np.concatenate(Y).astype(np.float32),
            np.concatenate(S).astype(np.float32),
            np.concatenate(G).astype(np.int64),
            np.concatenate(L).astype(np.int64))


if __name__ == '__main__':
    import time
    t0 = time.time()
    A, lab, names = rollout(30000, seed=0)
    print(f'{len(A)} telemetry rows in {time.time()-t0:.1f}s')
    print('names', names)
    print('regime counts', np.bincount(lab, minlength=4))
    for i, n in enumerate(names):
        print(f'  {n:8s} [{A[:,i].min():9.3f}, {A[:,i].max():9.3f}]  '
              f'mean {A[:,i].mean():8.3f}')
