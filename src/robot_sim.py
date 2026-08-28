"""
Multi-platform robot testbed on validated MuJoCo Menagerie models, with an
injected joint thermal network.

Rigid-body dynamics come from MuJoCo Menagerie (UR5e, Franka Panda, KUKA
iiwa14), whose inertial and actuator parameters come from the manufacturers.
None of that is ours. Two corrections are applied to make the models
physically honest for this study:

  * the Menagerie iiwa14 leaves actuator forcerange unset, i.e. unlimited, and
    an unconstrained multi-sine drives it to ~3500 N m, an order of magnitude
    past the real machine. Datasheet torque limits are imposed.
  * excitation amplitudes are velocity limited, A_i <= v_max / (2 pi f_i), so
    the commanded trajectory is one the arm can actually track instead of a
    permanent actuator saturation.

What we add on top is the thermal physics Menagerie does not model and that
this paper is about. Each joint carries the standard two-node lumped-parameter
thermal network used in drive and robot-joint identification:

    C_w dT_w/dt = k_cu tau^2 - (T_w - T_h) / R_wh
    C_h dT_h/dt = (T_w - T_h) / R_wh - (T_h - T_env) / R_ha

with a winding time constant of order a minute against a housing constant of
order twenty minutes, both against a 500 Hz control loop. That three-decade
separation is what makes the problem genuinely multi-physics: the mechanical
and thermal subsystems live on incommensurable time scales and are coupled
both ways. The reverse coupling is what bends the manifold:

  derating   a hot winding cannot pass rated current, so achievable torque
             falls as T_w approaches the limit and the closed-loop dynamics
             become temperature dependent;
  noise      PWM and commutation noise grow with speed and with heat, so the
             diffusion tensor V(s) is neither isotropic nor constant.

k_cu is set per joint from that joint's own torque limit so that continuous
operation at 35 % of rated torque settles near 55 K above ambient -- small
wrist motors therefore heat faster per newton-metre than the large shoulder
motors, as they do in reality.

Ground-truth regime labels are recorded but never shown to any model. They
exist only to ask, afterwards, whether the invariants recovered the physics.
"""

import os
import numpy as np

MENAGERIE = os.path.join(os.path.dirname(__file__), '..', 'data', 'mujoco_menagerie')

# datasheet limits: joint torque (N m) and joint speed (rad/s)
PLATFORMS = {
    'ur5e': dict(
        path='universal_robots_ur5e/ur5e.xml', n=6,
        tau_max=[150, 150, 150, 28, 28, 28],
        vel_max=[3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        flange='wrist_3_link'),
    'panda': dict(
        path='franka_emika_panda/panda.xml', n=7,
        tau_max=[87, 87, 87, 87, 12, 12, 12],
        vel_max=[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
        flange='hand'),
    'iiwa14': dict(
        path='kuka_iiwa_14/iiwa14.xml', n=7,
        tau_max=[320, 320, 176, 176, 110, 40, 40],
        vel_max=[1.48, 1.48, 1.75, 1.31, 2.27, 2.36, 2.36],
        flange='link7'),
}

R_WH = 0.5          # winding -> housing thermal resistance, K/W
R_HA = 1.2          # housing -> ambient thermal resistance, K/W
C_W = 80.0          # winding heat capacity, J/K  -> tau_w = C_W R_WH = 40 s
C_H = 900.0         # housing heat capacity, J/K  -> tau_h = C_H R_HA = 18 min
T_LIMIT = 105.0     # winding temperature at which torque is fully derated
T_KNEE = 70.0       # derating knee
DT_RATED = 100.0    # class-F winding rise at rated *continuous* torque
PEAK_TO_CONT = 0.35 # continuous / peak torque ratio for a servo joint
T_TRIP = 110.0      # winding temperature at which the drive faults out
T_RESET = 92.0      # temperature it must fall to before the drive re-engages
PAYLOAD_MAX = 5.0   # kg at the flange
PAYLOAD_RADIUS = 0.05    # m, payload modelled as a uniform sphere
T_RAMP = 5.0        # seconds to ramp excitation in from rest
T_WARMUP = 10.0     # seconds of telemetry discarded before recording


def k_cu_for(platform):
    """Per-joint ohmic gain, from motor physics rather than from a target.

    For a servo motor P_cu = R_phase I^2 and tau = K_t I, so the ohmic loss is
    (R_phase / K_t^2) tau^2: the gain is a fixed motor constant, not a free
    parameter. Motors are sized so that at rated continuous torque the winding
    sits at its insulation-class rise, so

        k_cu = DT_RATED / ((R_WH + R_HA) tau_rated^2)

    with DT_RATED = 100 K, a class-F rise.

    The rating that matters is the *continuous* torque, not the peak. MuJoCo's
    forcerange is the peak, and a servo joint's continuous rating is roughly a
    third of it; calibrating against peak torque understates the ohmic gain
    ninefold and leaves every platform thermally untouchable no matter how hard
    it is driven.

    An earlier version calibrated k_cu from a probe run so that every platform
    reached the same temperature. That is backwards, and it was actively
    harmful: the iiwa14 draws only ~0.19 of rated torque under this excitation,
    so forcing it to a 65 K rise demanded a gain two orders of magnitude too
    large, and the moment a torque transient hit the limit the winding
    integrated to 760 K and the servo went unstable. Deriving the gain from the
    rating instead bounds the ohmic term at DT_RATED / (R_WH + R_HA) by
    construction.

    The consequence is that platforms differ in thermal headroom under the same
    duty, which is the physical truth: the iiwa14 is simply oversized for this
    task and stays near ambient, while the Panda runs hot. That contrast is
    reported rather than engineered away.
    """
    tau_peak = np.asarray(PLATFORMS[platform]['tau_max'], dtype=float)
    tau_cont = PEAK_TO_CONT * tau_peak
    return DT_RATED / ((R_WH + R_HA) * tau_cont ** 2)


def _excitation(rng, nj, half, cfg, duty, n_freq=5):
    """Band-limited multi-sine with position and speed budgets shared across
    components. Peak speed of a sum of sinusoids is sum_i 2 pi f_i A_i, so
    granting each component the full budget overshoots it n_freq-fold.

    The band is 0.2-1.5 Hz, the rate of an industrial pick-and-place cycle. A
    slower 0.05-0.6 Hz band is dynamically fine but draws only a fifth of rated
    torque, so no platform ever approaches its thermal limit and the derating
    and trip regimes never occur. Under the velocity budget the amplitude falls
    as 1/f, so peak acceleration -- and hence torque -- grows linearly with the
    band while speed stays inside the datasheet limit."""
    f = rng.uniform(0.2, 1.5, (nj, n_freq))
    ph = rng.uniform(0, 2 * np.pi, (nj, n_freq))
    a = rng.uniform(0.3, 1.0, (nj, n_freq))
    a /= a.sum(1, keepdims=True)
    vmax = np.asarray(cfg['vel_max'])[:, None]
    a = np.minimum(0.35 * half[:, None] * a, duty * vmax * a / (2 * np.pi * f))
    return a, f, ph


def attach_payload(m, platform, payload):
    """Attach a payload to the named flange body, mass *and* inertia.

    Two things go wrong if this is done casually. The payload must land on the
    flange, not on whatever body happens to be last in the model: for the Panda
    that is `right_finger`, a 15 g gripper tip, and hanging 5 kg off it is a
    333-fold mass increase on a body with 2e-6 kg m^2 of inertia. And the
    inertia has to move with the mass -- adding kilograms while leaving the
    inertia tensor untouched creates a heavy body that is trivially easy to
    spin, which is not a rigid body at all and which drove the iiwa14 servo
    unstable at 80 rad/s.

    The payload is modelled as a uniform sphere at the flange origin, so
    I += (2/5) m r^2 on each principal axis.
    """
    import mujoco
    if payload <= 0:
        return
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PLATFORMS[platform]['flange'])
    if bid < 0:
        raise ValueError(f'no flange body for {platform}')
    m.body_mass[bid] += payload
    m.body_inertia[bid] += 0.4 * payload * PAYLOAD_RADIUS ** 2


def _load(platform):
    import mujoco
    cfg = PLATFORMS[platform]
    m = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, cfg['path']))
    nj = cfg['n']
    # impose datasheet torque limits where the model leaves them unset
    fr = m.actuator_forcerange[:nj]
    unset = (fr[:, 0] == 0) & (fr[:, 1] == 0)
    if unset.any():
        tm = np.asarray(cfg['tau_max'], dtype=float)
        m.actuator_forcerange[:nj][unset] = np.stack(
            [-tm[unset], tm[unset]], -1)
        m.actuator_forcelimited[:nj][unset] = 1
    return m, nj, cfg


def rollout(platform='ur5e', seconds=180.0, seed=0, telemetry_hz=50.0,
            duty=None, t_env=None, payload=None):
    """One episode of one robot. Returns (rows, labels, column names)."""
    import mujoco
    m, nj, cfg = _load(platform)
    d = mujoco.MjData(m)
    rng = np.random.RandomState(seed)

    dt = m.opt.timestep
    decim = max(1, int(round(1.0 / (telemetry_hz * dt))))
    n_steps = int(seconds / dt)

    lo, hi = m.jnt_range[:nj, 0].copy(), m.jnt_range[:nj, 1].copy()
    bad = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
    lo[bad], hi[bad] = -2.5, 2.5
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)

    duty = rng.uniform(0.25, 1.0) if duty is None else duty
    t_env = rng.uniform(18.0, 34.0) if t_env is None else t_env
    payload = rng.uniform(0.0, PAYLOAD_MAX) if payload is None else payload
    attach_payload(m, platform, payload)

    a, f, ph = _excitation(rng, nj, half, cfg, duty)
    k_cu = k_cu_for(platform)
    Tw = np.full(nj, t_env) + rng.randn(nj) * 0.5
    Th = np.full(nj, t_env) + rng.randn(nj) * 0.5
    tripped = np.zeros(nj, dtype=bool)
    hold_pos = np.zeros(nj)

    # Start the arm *on* the trajectory and ramp the excitation in. Dropping
    # the arm at a random pose and immediately commanding a multi-sine is a
    # step input to a stiff position servo: the iiwa14 rang at 73 rad/s, 30x
    # its datasheet limit, purely as a startup artefact.
    def cmd(t, ramp=True):
        g = min(1.0, t / T_RAMP) if ramp else 1.0
        return np.clip(mid + g * (a * np.sin(2 * np.pi * f * t + ph)).sum(1), lo, hi)

    d.qpos[:nj] = cmd(0.0)
    mujoco.mj_forward(m, d)

    n_warm = int(T_WARMUP / dt)
    rows, labs = [], []
    for i in range(n_steps + n_warm):
        t = i * dt
        u = cmd(t)

        # Derating: a hot winding cannot pass rated current. Applied as reduced
        # command authority about the current pose, which is how a drive in
        # current limit actually behaves.
        derate = np.clip((T_LIMIT - Tw) / (T_LIMIT - T_KNEE), 0.15, 1.0)

        # Thermal trip with hysteresis. Derating alone cannot bound the
        # temperature, because holding a loaded arm against gravity costs
        # torque even at zero command authority; without a cutout the UR5e
        # shoulder reaches ~180 C. Real drives fault out and let a holding
        # brake take the load, so the ohmic source goes to zero until the
        # winding has cooled to T_RESET. The resulting switching boundary is
        # sharp, hysteretic and entirely physical.
        newly = (~tripped) & (Tw > T_TRIP)
        tripped = (tripped & (Tw > T_RESET)) | (Tw > T_TRIP)
        hold_pos = np.where(newly, d.qpos[:nj], hold_pos)
        u = d.qpos[:nj] + derate * (u - d.qpos[:nj])
        # A tripped joint is held at the pose it faulted at. Steering the
        # target to the *current* position each step instead removes the
        # restoring force entirely, which is a release rather than a brake: the
        # iiwa14's heavy links then free-fell, the surviving joints saturated
        # trying to catch them, and the winding ran to 500 C. Latching the pose
        # is what a holding brake actually does.
        u = np.where(tripped, hold_pos, u)

        spd = np.abs(d.qvel[:nj])
        sd = 1e-4 * (1.0 + 3.0 * spd / (1.0 + spd)
                     + 2.0 * np.clip((Tw - 60.0) / 50.0, 0, 1))
        d.ctrl[:nj] = np.clip(u + rng.randn(nj) * sd, lo, hi)
        mujoco.mj_step(m, d)

        tau = d.qfrc_actuator[:nj].copy()
        tau_heat = np.where(tripped, 0.0, tau)      # brake carries a tripped joint
        Tw = Tw + dt * (k_cu * tau_heat ** 2 - (Tw - Th) / R_WH) / C_W
        Th = Th + dt * ((Tw - Th) / R_WH - (Th - t_env) / R_HA) / C_H

        if i >= n_warm and i % decim == 0:
            rows.append(np.concatenate([d.qpos[:nj], d.qvel[:nj], tau, Tw, Th,
                                        [t_env, payload]]))
            # Regimes are a property of the operating point, not of the
            # episode. Payload is fixed within an episode, so keying a regime
            # on it just relabels episodes; it is kept as a continuous
            # covariate instead. The 2x2 of speed and thermal state is what
            # actually varies along a trajectory.
            v = (np.abs(d.qvel[:nj]) / np.asarray(cfg['vel_max'])).max()
            if tripped.any():
                labs.append(4)                       # drive faulted, brake holding
            else:
                labs.append((2 if Tw.max() > T_KNEE else 0) + (1 if v > 0.30 else 0))

    rows = np.array(rows, dtype=np.float32)
    # Guard: a diverged episode is worse than a missing one, because it looks
    # like data. Anything an order of magnitude past the datasheet envelope is
    # an integration failure, not a duty cycle.
    dq = rows[:, nj:2 * nj]
    if np.abs(dq).max() > 3.0 * max(cfg['vel_max']):
        raise RuntimeError(f'{platform} seed {seed} diverged: '
                           f'max|dq|={np.abs(dq).max():.1f} rad/s')
    return rows, np.array(labs), state_cols(platform)


def state_cols(platform):
    nj = PLATFORMS[platform]['n']
    return ([f'q{j}' for j in range(nj)] + [f'dq{j}' for j in range(nj)]
            + [f'tau{j}' for j in range(nj)] + [f'Tw{j}' for j in range(nj)]
            + [f'Th{j}' for j in range(nj)] + ['Tenv', 'payload'])


def _episode(args):
    platform, seed, seconds, hz = args
    A, lab, names = rollout(platform, seconds=seconds, seed=seed, telemetry_hz=hz)
    return A, lab, seed


def build_dataset(platform='ur5e', n_ep=40, seconds=180.0, seed0=0,
                  horizon=10, telemetry_hz=50.0, workers=10):
    """Episodes -> (X, Y, S, G, L).

    Task: a short-horizon forward model. From the operating state and a causal
    summary of its recent history, predict joint velocities and winding
    temperatures `horizon` telemetry steps ahead -- the mechanical and thermal
    subsystems at once, which is the point.
    """
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(platform, seed0 + e, seconds, telemetry_hz) for e in range(n_ep)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_episode, jobs))

    nj = PLATFORMS[platform]['n']
    X, Y, S, G, L = [], [], [], [], []
    for st, lab, seed in out:
        feats = [st]
        for span in (10, 50, 250):
            al = 2.0 / (span + 1)
            ew = np.empty_like(st)
            acc = st[0].copy()
            for i in range(len(st)):
                acc = al * st[i] + (1 - al) * acc
                ew[i] = acc
            feats.append(ew)
            feats.append(st - ew)
        F = np.concatenate(feats, 1)
        tgt = np.concatenate([st[:, nj:2 * nj], st[:, 3 * nj:4 * nj]], 1)
        n = len(st) - horizon
        X.append(F[:n]); Y.append(tgt[horizon:horizon + n])
        S.append(st[:n]); L.append(lab[:n]); G.append(np.full(n, seed))
    return (np.concatenate(X).astype(np.float32),
            np.concatenate(Y).astype(np.float32),
            np.concatenate(S).astype(np.float32),
            np.concatenate(G).astype(np.int64),
            np.concatenate(L).astype(np.int64))


if __name__ == '__main__':
    import time, sys
    plat = sys.argv[1] if len(sys.argv) > 1 else 'ur5e'
    cfg = PLATFORMS[plat]
    t0 = time.time()
    A, lab, names = rollout(plat, seconds=180.0, seed=0)
    nj = cfg['n']
    print(f'{plat}: {len(A)} rows in {time.time()-t0:.1f}s   '
          f'regimes {np.bincount(lab, minlength=5)}')
    for grp, sl, lim in [('q', slice(0, nj), None),
                         ('dq', slice(nj, 2 * nj), cfg['vel_max']),
                         ('tau', slice(2 * nj, 3 * nj), cfg['tau_max']),
                         ('Tw', slice(3 * nj, 4 * nj), None),
                         ('Th', slice(4 * nj, 5 * nj), None)]:
        B = A[:, sl]
        extra = ''
        if lim is not None:
            r = np.abs(B) / np.asarray(lim)
            extra = (f'  p99.9/limit {np.quantile(r, 0.999):.2f}'
                     f'  max/limit {r.max():.2f}'
                     f'  frac_sat {float((r > 0.99).mean()):.3f}')
        print(f'  {grp:4s} [{B.min():9.3f}, {B.max():9.3f}] mean {B.mean():8.3f}{extra}')
