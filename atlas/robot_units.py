"""
Robot UNITS: many individual machines of the same class.

The published robot testbed draws payload, duty and ambient from the episode
seed, so every episode is effectively a different machine and there is no such
thing as 'the same unit observed twice'. That is fine for a routing study and
useless for this one. Here the two are deliberately separated:

    UNIT IDENTITY   fixed physical parameters that belong to the machine and
                    persist across everything it ever does -- payload it
                    carries, thermal resistances, winding resistance, bearing
                    friction, servo gain. Drawn once per unit.

    WORKLOAD        excitation seed, duty cycle, ambient temperature. Drawn
                    fresh per episode.

This separation is what makes the central claim falsifiable. If an atlas built
from a unit's early episodes retrieves that same unit from LATER episodes run
under a DIFFERENT workload, the atlas has encoded the machine and not the task.
If it fails, the whole idea is a workload detector wearing a costume.

The ground-truth identity vector is recorded and never shown to any model. It
exists so that afterwards we can ask whether the recovered unit code decodes
the physics -- an experiment no public dataset permits, because on real fleets
nobody knows the true bearing friction of unit 47.
"""

import os
import sys
import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import robot_sim as rs   # noqa: E402  PLATFORMS, k_cu_for, _excitation, constants

# ---------------------------------------------------------------------------
# unit identity
# ---------------------------------------------------------------------------
# Each entry is (name, low, high) in MULTIPLIER space unless stated otherwise.
# Ranges are wide enough to matter physically and narrow enough that every unit
# is still a working machine: a robot that trips continuously carries no
# information about its bearings.
IDENTITY = [
    ('payload',    0.0, 5.0),    # kg at the flange (absolute, not a multiplier)
    ('r_wh',       0.7, 1.6),    # winding -> housing resistance: mounting, paste
    ('r_ha',       0.7, 2.0),    # housing -> ambient: fouled fins, blocked fan
    ('k_cu',       0.8, 1.8),    # winding resistance: insulation age, hot copper
    ('damp',       0.5, 2.5),    # bearing friction / lubricant condition
    ('gain',       0.88, 1.12),  # servo proportional gain: tuning drift
    ('skew',       0.0, 1.0),    # how unevenly the above load onto the joints
]
IDENTITY_NAMES = [k for k, _, _ in IDENTITY]


def sample_identity(rng):
    return np.array([lo + (hi - lo) * rng.random() for _, lo, hi in IDENTITY])


def _joint_weights(nj, skew, rng):
    """Per-joint modulation of the identity parameters.

    A real machine does not wear evenly: one bearing goes first, one winding
    runs hotter. `skew` interpolates between a uniform machine (every joint
    identical) and a strongly non-uniform one. The pattern is fixed by the
    unit's own seed so it is part of the unit's identity, not noise.
    """
    w = 1.0 + skew * rng.uniform(-0.6, 0.6, nj)
    return np.clip(w, 0.25, 2.5)


def rollout_unit(platform, ident, unit_seed, ep_seed, seconds=120.0,
                 telemetry_hz=50.0, duty=None, t_env=None):
    """One episode of one specific machine.

    Mirrors robot_sim.rollout -- same excitation, derating, hysteretic trip and
    two-node thermal network -- with the unit's physical parameters substituted
    for the module-level constants.
    """
    import mujoco
    m, nj, cfg = rs._load(platform)
    d = mujoco.MjData(m)
    rng = np.random.RandomState(ep_seed)
    urng = np.random.RandomState(unit_seed)

    payload, r_wh_m, r_ha_m, k_cu_m, damp_m, gain_m, skew = ident
    jw = _joint_weights(nj, skew, urng)

    # --- apply the unit's identity to the model -----------------------------
    rs.attach_payload(m, platform, float(payload))
    base_damp = m.dof_damping[:nj].copy()
    # Menagerie leaves damping at 0 on some models; a multiplier on zero is
    # still zero, so add a small physical floor before scaling, otherwise the
    # bearing-condition parameter would be silently unidentifiable.
    floor = 0.05 * np.asarray(cfg['tau_max'], dtype=float) / np.asarray(cfg['vel_max'])
    m.dof_damping[:nj] = (np.maximum(base_damp, floor) * damp_m * jw)
    if m.actuator_gainprm.shape[0] >= nj:
        m.actuator_gainprm[:nj, 0] *= gain_m
        # position actuators carry -kp in biasprm[:,1]; scaling gain alone
        # would turn the servo into a mismatched PD and change the *class*
        # dynamics rather than this unit's tuning
        m.actuator_biasprm[:nj, 1] *= gain_m

    R_WH = rs.R_WH * r_wh_m * jw
    R_HA = rs.R_HA * r_ha_m * jw
    k_cu = rs.k_cu_for(platform) * k_cu_m * jw

    dt = m.opt.timestep
    decim = max(1, int(round(1.0 / (telemetry_hz * dt))))
    n_steps = int(seconds / dt)

    lo, hi = m.jnt_range[:nj, 0].copy(), m.jnt_range[:nj, 1].copy()
    bad = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
    lo[bad], hi[bad] = -2.5, 2.5
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)

    duty = rng.uniform(0.25, 1.0) if duty is None else duty
    t_env = rng.uniform(18.0, 34.0) if t_env is None else t_env
    a, f, ph = rs._excitation(rng, nj, half, cfg, duty)

    Tw = np.full(nj, t_env) + rng.randn(nj) * 0.5
    Th = np.full(nj, t_env) + rng.randn(nj) * 0.5
    tripped = np.zeros(nj, dtype=bool)
    hold_pos = np.zeros(nj)

    n_warm = int(rs.T_WARMUP / dt)
    n_tot = n_steps + n_warm

    # The excitation and the commutation noise are deterministic functions of
    # time and of the episode seed, so both are built once as whole
    # trajectories. Evaluating a (nj x 5) sine and drawing nj gaussians inside
    # the step loop cost ~2 ms per step against MuJoCo's own ~50 us, i.e. 98 %
    # of the runtime went on numpy call overhead rather than on physics.
    # The servo runs at 100 Hz and telemetry is logged at 50 Hz, both realistic
    # for an industrial arm. Between servo updates MuJoCo integrates in C via
    # nstep, so the Python loop runs once per control period instead of once per
    # 2 ms physics step. The thermal state advances on the control period too,
    # which is exact to O(sub*dt / tau_w) = 2.5e-4 given a 40 s winding constant.
    sub = max(1, int(round(1.0 / (100.0 * dt))))
    n_blk = int(np.ceil(n_tot / sub))
    n_tot = n_blk * sub
    blk_dt = sub * dt
    decim_blk = max(1, int(round(1.0 / (telemetry_hz * blk_dt))))
    n_warm_blk = int(np.ceil(n_warm / sub))

    tb = np.arange(n_blk, dtype=np.float64) * blk_dt
    gr = np.minimum(1.0, tb / rs.T_RAMP)[:, None]
    osc = np.einsum('jk,tjk->tj', a, np.sin(2 * np.pi * f[None] * tb[:, None, None]
                                            + ph[None]))
    CMD = np.clip(mid + gr * osc, lo, hi)
    NOISE = rng.randn(n_blk, nj)

    d.qpos[:nj] = CMD[0]
    mujoco.mj_forward(m, d)

    vmax = np.asarray(cfg['vel_max'])
    inv_span = 1.0 / (rs.T_LIMIT - rs.T_KNEE)
    rows, labs = [], []
    qp = d.qpos[:nj]
    qv = d.qvel[:nj]
    for i in range(n_blk):
        derate = np.clip((rs.T_LIMIT - Tw) * inv_span, 0.15, 1.0)
        newly = (~tripped) & (Tw > rs.T_TRIP)
        tripped = (tripped & (Tw > rs.T_RESET)) | (Tw > rs.T_TRIP)
        if newly.any():
            hold_pos = np.where(newly, qp, hold_pos)
        u = qp + derate * (CMD[i] - qp)
        if tripped.any():
            u = np.where(tripped, hold_pos, u)

        spd = np.abs(qv)
        sd = 1e-4 * (1.0 + 3.0 * spd / (1.0 + spd)
                     + 2.0 * np.clip((Tw - 60.0) * 0.02, 0, 1))
        d.ctrl[:nj] = np.clip(u + NOISE[i] * sd, lo, hi)
        mujoco.mj_step(m, d, nstep=sub)

        tau = d.qfrc_actuator[:nj]
        tau_heat = np.where(tripped, 0.0, tau)
        dTw = (k_cu * tau_heat * tau_heat - (Tw - Th) / R_WH) / rs.C_W
        Th = Th + blk_dt * ((Tw - Th) / R_WH - (Th - t_env) / R_HA) / rs.C_H
        Tw = Tw + blk_dt * dTw

        if i >= n_warm_blk and i % decim_blk == 0:
            rows.append(np.concatenate([qp, qv, tau, Tw, Th, [t_env]]))
            v = (np.abs(qv) / vmax).max()
            if tripped.any():
                labs.append(4)
            else:
                labs.append((2 if Tw.max() > rs.T_KNEE else 0) + (1 if v > 0.30 else 0))

    rows = np.array(rows, dtype=np.float32)
    dq = rows[:, nj:2 * nj]
    if not np.isfinite(rows).all() or np.abs(dq).max() > 3.0 * max(cfg['vel_max']):
        raise RuntimeError(f'{platform} unit {unit_seed} ep {ep_seed} diverged')
    return rows, np.array(labs)


def channel_names(platform):
    """NOTE: payload is deliberately NOT a channel. It is part of the unit's
    ground-truth identity, and exposing it would let the decode experiment read
    the answer off the input."""
    nj = rs.PLATFORMS[platform]['n']
    return ([f'q{j}' for j in range(nj)] + [f'dq{j}' for j in range(nj)]
            + [f'tau{j}' for j in range(nj)] + [f'Tw{j}' for j in range(nj)]
            + [f'Th{j}' for j in range(nj)] + ['Tenv'])


def _one(args):
    platform, ui, ident, ep, seconds, hz = args
    # Workload is drawn here, explicitly and recorded, rather than inside the
    # rollout. Without it there is no way to run the control that matters: an
    # atlas that merely detects how hard a machine was driven would score well
    # on identity too, because a unit's episodes share nothing else. Decoding
    # workload from the same features is what separates the two.
    wrng = np.random.RandomState(500000 + 977 * ui + ep)
    duty = float(wrng.uniform(0.25, 1.0))
    t_env = float(wrng.uniform(18.0, 34.0))
    try:
        rows, labs = rollout_unit(platform, ident, unit_seed=1000 + ui,
                                  ep_seed=100000 + 137 * ui + ep,
                                  seconds=seconds, telemetry_hz=hz,
                                  duty=duty, t_env=t_env)
        return (ui, ep, rows, labs, (duty, t_env), None)
    except Exception as e:                      # a diverged episode is dropped,
        return (ui, ep, None, None, (duty, t_env), str(e))   # never silently kept


def build_units(platform='ur5e', n_units=60, n_ep=6, seconds=120.0,
                telemetry_hz=50.0, workers=4, seed=0, out=None):
    """Simulate a fleet. Returns dict with per-unit episode arrays + identities."""
    from concurrent.futures import ProcessPoolExecutor
    import time
    rng = np.random.default_rng(seed)
    idents = np.array([sample_identity(rng) for _ in range(n_units)])
    jobs = [(platform, ui, idents[ui], ep, seconds, telemetry_hz)
            for ui in range(n_units) for ep in range(n_ep)]
    t0 = time.time()
    eps = {}
    work = {}
    fails = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (ui, ep, rows, labs, wl, err) in enumerate(
                ex.map(_one, jobs, chunksize=1)):
            if rows is None:
                fails += 1
                continue
            eps.setdefault(ui, []).append((ep, rows, labs))
            work[(ui, ep)] = wl
            if (k + 1) % 50 == 0:
                print(f'    {platform} {k+1}/{len(jobs)} eps  '
                      f'[{time.time()-t0:.0f}s, {fails} dropped]', flush=True)
    print(f'  {platform}: {len(eps)} units, {sum(len(v) for v in eps.values())} '
          f'episodes, {fails} dropped, {time.time()-t0:.0f}s', flush=True)
    res = dict(platform=platform, idents=idents, channels=channel_names(platform),
               identity_names=IDENTITY_NAMES, episodes=eps)
    if out:
        np.savez_compressed(
            out,
            idents=idents,
            channels=np.array(channel_names(platform)),
            identity_names=np.array(IDENTITY_NAMES),
            unit_ids=np.array([ui for ui in sorted(eps) for _ in eps[ui]]),
            ep_ids=np.array([e for ui in sorted(eps) for e, _, _ in eps[ui]]),
            workload=np.array([work[(ui, e)] for ui in sorted(eps)
                               for e, _, _ in eps[ui]]),
            workload_names=np.array(['duty', 't_env']),
            **{f'X_{ui}_{e}': r for ui in sorted(eps) for e, r, _ in eps[ui]},
            **{f'L_{ui}_{e}': l for ui in sorted(eps) for e, _, l in eps[ui]})
        print(f'  wrote {out}', flush=True)
    return res


if __name__ == '__main__':
    import time
    plat = sys.argv[1] if len(sys.argv) > 1 else 'ur5e'
    rng = np.random.default_rng(0)
    ident = sample_identity(rng)
    print(f'{plat} identity:', dict(zip(IDENTITY_NAMES, np.round(ident, 3))))
    t0 = time.time()
    rows, labs = rollout_unit(plat, ident, 1000, 200000, seconds=30.0)
    print(f'  {rows.shape} rows in {time.time()-t0:.1f}s  '
          f'regimes {np.bincount(labs, minlength=5)}')
    nj = rs.PLATFORMS[plat]['n']
    print(f'  Tw range [{rows[:, 3*nj:4*nj].min():.1f}, {rows[:, 3*nj:4*nj].max():.1f}] C')
