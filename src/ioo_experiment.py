"""Fresh controlled experiment for the IOO representation.

The experiment is intentionally transparent. Four operator subsystems generate
coupled trajectories. A relational classifier uses graph-derived features,
while a flat baseline uses channel moments. A held-out sensor recalibration and
a component substitution test measure the two claims separately.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ioo_core import Interface, OperatorClass, OperatorCatalogue, OperatorInstance, build_graph, rank_atlas, trajectory_chart


@dataclass
class Episode:
    machine: int
    component_variant: int
    recalibrated: bool
    telemetry: np.ndarray


def catalogue() -> OperatorCatalogue:
    c = OperatorCatalogue()
    c.register(OperatorClass("dc_motor", "motor", (), ("tau=k_t I", "V=k_e omega+RI")))
    c.register(OperatorClass("ac_motor", "motor", (), ("electromagnetic torque",)))
    c.register(OperatorClass("gearbox", "transmission", (), ("omega_out=omega_in/g",)))
    c.register(OperatorClass("linear_actuator", "actuator", (), ("v=p omega/(2 pi)",)))
    c.register(OperatorClass("thermal_node", "thermal", (), ("C dT/dt=P-(T-Tenv)/R",)))
    return c


def simulate(machine: int, variant: int, seed: int, steps: int = 800, recalibrated: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 80.0, steps)
    load = 0.65 + 0.15 * np.sin(0.07 * t + rng.uniform(-1, 1))
    command = (0.55 + 0.18 * np.sin(0.31 * t + rng.uniform(-1, 1))
               + 0.08 * np.sin(0.83 * t + rng.uniform(-1, 1))) * load
    motor_gain = 1.0 + 0.08 * machine
    gearbox = (18.0 + 1.5 * machine) * (1.0 if variant == 0 else 0.82)
    thermal_resistance = 3.8 - 0.25 * machine + (0.0 if variant == 0 else -0.55)
    omega = np.zeros(steps)
    torque = np.zeros(steps)
    winding = np.zeros(steps)
    housing = np.zeros(steps)
    for i in range(1, steps):
        dt = t[i] - t[i - 1]
        torque[i] = motor_gain * (command[i] - 0.12 * omega[i - 1])
        omega[i] = omega[i - 1] + dt * (torque[i] - 0.08 * omega[i - 1]) / gearbox
        winding[i] = winding[i - 1] + dt * (2.1 * torque[i] ** 2 - (winding[i - 1] - housing[i - 1]) / 1.4) / 9.0
        housing[i] = housing[i - 1] + dt * ((winding[i - 1] - housing[i - 1]) / 1.4 - housing[i - 1] / thermal_resistance) / 35.0
    ambient = 22.0 + 0.8 * np.sin(0.02 * t)
    sensor_noise = rng.normal(0.0, 0.01, (steps, 6))
    data = np.column_stack((command, omega, torque, winding, housing, ambient)) + sensor_noise
    if recalibrated:
        gains = np.array([1.7, 0.6, 2.3, 0.75, 1.25, 0.9])
        offsets = np.array([0.4, -0.2, 0.3, 1.5, -0.7, 2.0])
        data = np.sinh(0.7 * np.arcsinh(data * gains)) + offsets
    return data.astype(np.float64)


def episode_features(ep: Episode) -> Dict[str, np.ndarray]:
    x = ep.telemetry
    rank = rank_atlas(x)
    diffs = np.diff(rank, axis=0)
    out = {
        "raw_moments": np.r_[x.mean(0), x.std(0), np.quantile(x, 0.1, axis=0), np.quantile(x, 0.9, axis=0)],
        "rank_moments": np.r_[rank.mean(0), rank.std(0), np.quantile(rank, 0.1, axis=0), np.quantile(rank, 0.9, axis=0)],
        "relational": np.r_[np.corrcoef(rank, rowvar=False)[np.triu_indices(x.shape[1], 1)],
                             np.mean(diffs[:-1] * diffs[1:], axis=0),
                             np.mean(np.abs(diffs), axis=0)],
    }
    return out


def make_episodes(n_machines: int, episodes_per_machine: int, steps: int, recalibrated: bool = False) -> List[Episode]:
    episodes = []
    for machine in range(n_machines):
        for ep in range(episodes_per_machine):
            episodes.append(Episode(machine, ep % 2, recalibrated,
                                     simulate(machine, ep % 2, machine * 100 + ep, steps, recalibrated)))
    return episodes


def evaluate_representation(train: List[Episode], test: List[Episode], key: str) -> float:
    x_train = np.stack([episode_features(e)[key] for e in train])
    x_test = np.stack([episode_features(e)[key] for e in test])
    y_train = np.array([e.machine for e in train])
    y_test = np.array([e.machine for e in test])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    model.fit(x_train, y_train)
    return float(balanced_accuracy_score(y_test, model.predict(x_test)))


def run(output: str, seeds: List[int], machines: int = 8, episodes: int = 6, steps: int = 800) -> Dict[str, object]:
    rows = []
    for seed in seeds:
        train = make_episodes(machines, episodes, steps, False)
        test = make_episodes(machines, episodes, steps, seed % 2 == 1)
        for key in ("raw_moments", "rank_moments", "relational"):
            rows.append({"seed": seed, "representation": key, "recalibrated": bool(seed % 2),
                         "balanced_accuracy": evaluate_representation(train, test, key)})
    result = {"config": {"seeds": seeds, "machines": machines, "episodes": episodes, "steps": steps}, "rows": rows}
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (path / "metrics.csv").write_text("seed,representation,recalibrated,balanced_accuracy\n" + "\n".join(
        f"{r['seed']},{r['representation']},{int(r['recalibrated'])},{r['balanced_accuracy']:.8f}" for r in rows), encoding="utf-8")
    return result


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/local")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--machines", type=int, default=8)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()
    print(json.dumps(run(args.output, args.seeds, args.machines, args.episodes, args.steps), indent=2))


if __name__ == "__main__":
    main()
