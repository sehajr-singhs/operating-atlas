"""Core data structures for the fresh Index of Operations implementation.

The graph is deliberately typed. Physical interfaces, statistical couplings,
and temporal adjacency are different edge relations and are never silently
merged into one claim of causality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import Isomap


@dataclass(frozen=True)
class Port:
    name: str
    direction: str
    quantity: str
    unit: str


@dataclass(frozen=True)
class OperatorClass:
    name: str
    family: str
    ports: Tuple[Port, ...]
    equations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorInstance:
    instance_id: str
    operator_class: str
    variables: Tuple[str, ...]
    metadata: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Interface:
    source: str
    target: str
    relation: str
    parameters: Mapping[str, float] = field(default_factory=dict)


@dataclass
class IOOGraph:
    """A single graph snapshot or graph sequence.

    `physical_edges` are supplied by engineering structure. `statistical_edges`
    are estimated from telemetry. `temporal_edges` connect consecutive
    snapshots. The model may consume all three, but analyses must preserve the
    provenance of each edge.
    """

    node_ids: List[str]
    node_types: Dict[str, str]
    node_features: np.ndarray
    physical_edges: List[Tuple[str, str, str, Dict[str, float]]]
    statistical_edges: List[Tuple[str, str, str, Dict[str, float]]]
    temporal_edges: List[Tuple[str, str, str, Dict[str, float]]]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def all_edges(self):
        return self.physical_edges + self.statistical_edges + self.temporal_edges


class OperatorCatalogue:
    def __init__(self) -> None:
        self._classes: Dict[str, OperatorClass] = {}

    def register(self, operator: OperatorClass) -> None:
        if operator.name in self._classes:
            raise ValueError(f"operator class already registered: {operator.name}")
        self._classes[operator.name] = operator

    def get(self, name: str) -> OperatorClass:
        return self._classes[name]

    def names(self) -> Tuple[str, ...]:
        return tuple(self._classes)


def standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("telemetry must have shape (time, variables)")
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    clean = np.where(np.isfinite(x), x, mean)
    return (clean - mean) / scale, mean, scale


def rolling_coupling(x: np.ndarray, window: int = 128) -> np.ndarray:
    """Return absolute lag-zero Pearson coupling at each window midpoint."""
    x = np.asarray(x, dtype=np.float64)
    if window < 4 or window > len(x):
        raise ValueError("window must be between 4 and the number of rows")
    out = []
    for start in range(0, len(x) - window + 1, max(1, window // 4)):
        corr = np.corrcoef(x[start:start + window], rowvar=False)
        out.append(np.nan_to_num(corr, nan=0.0))
    return np.asarray(out)


def build_edges_from_coupling(
    node_ids: Sequence[str], coupling: np.ndarray, threshold: float = 0.65
) -> List[Tuple[str, str, str, Dict[str, float]]]:
    edges = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            weight = float(abs(coupling[i, j]))
            if weight >= threshold:
                edges.extend([
                    (node_ids[i], node_ids[j], "statistical_coupling", {"weight": weight}),
                    (node_ids[j], node_ids[i], "statistical_coupling", {"weight": weight}),
                ])
    return edges


def trajectory_chart(telemetry: np.ndarray, components: int = 3, neighbours: int = 12) -> np.ndarray:
    """Project high-dimensional machine states for visualization only."""
    x, _, _ = standardize(telemetry)
    components = min(components, x.shape[1], max(1, len(x) - 1))
    if components == 1:
        return x[:, :1]
    model = Isomap(n_neighbors=min(neighbours, len(x) - 1), n_components=components)
    return model.fit_transform(x)


def rank_atlas(telemetry: np.ndarray) -> np.ndarray:
    """Channelwise rank coordinates, invariant to strictly increasing warps."""
    x = np.asarray(telemetry, dtype=np.float64)
    ranks = np.empty_like(x)
    for j in range(x.shape[1]):
        order = np.argsort(np.argsort(x[:, j], kind="stable"), kind="stable")
        ranks[:, j] = (order + 0.5) / len(x)
    return ranks


def build_graph(
    telemetry: np.ndarray,
    instances: Sequence[OperatorInstance],
    interfaces: Sequence[Interface],
    variable_groups: Mapping[str, Sequence[int]],
    threshold: float = 0.65,
) -> IOOGraph:
    """Construct an IOO graph from one aligned telemetry window."""
    x, _, _ = standardize(telemetry)
    node_ids = [i.instance_id for i in instances]
    node_types = {i.instance_id: i.operator_class for i in instances}
    features = []
    for instance in instances:
        cols = variable_groups[instance.instance_id]
        features.append(np.nanmean(x[:, list(cols)], axis=0))
    width = max(len(row) for row in features)
    node_features = np.zeros((len(features), width), dtype=np.float64)
    for row, values in enumerate(features):
        node_features[row, :len(values)] = values

    physical = []
    for interface in interfaces:
        physical.append((interface.source, interface.target, interface.relation,
                         dict(interface.parameters)))
    coupling = np.corrcoef(x, rowvar=False)
    variable_ids = [f"v{i}" for i in range(x.shape[1])]
    statistical = build_edges_from_coupling(variable_ids, coupling, threshold)
    temporal = [(node_ids[i], node_ids[i], "temporal", {"lag": 1.0})
                for i in range(len(node_ids))]
    return IOOGraph(node_ids, node_types, node_features, physical, statistical,
                    temporal, {"variables": variable_ids, "standardized": True})
