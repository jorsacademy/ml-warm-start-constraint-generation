"""Euclidean TSP domain, canonical subtour cuts, and independent exact oracles."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

from warmcg.utils import finite_float, integer, string

SUPPORTED_REGIMES = (
    "uniform",
    "clustered",
    "strongly_clustered",
    "ring",
    "grid_jitter",
    "anisotropic",
    "outlier",
    "two_cluster_bridge",
)


@dataclass(frozen=True)
class TSPInstance:
    """A finite complete undirected Euclidean TSP instance."""

    coordinates: tuple[tuple[float, float], ...]
    instance_id: str = "instance"
    regime: str = "unspecified"
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.coordinates) < 5:
            raise ValueError("a TSP instance must contain at least five nodes")
        normalized: list[tuple[float, float]] = []
        for index, coordinate in enumerate(self.coordinates):
            if len(coordinate) != 2:
                raise ValueError(f"coordinate {index} must have two components")
            x = finite_float(coordinate[0], name=f"coordinate[{index}].x")
            y = finite_float(coordinate[1], name=f"coordinate[{index}].y")
            normalized.append((x, y))
        if len(set(normalized)) != len(normalized):
            raise ValueError("coordinates must be pairwise distinct")
        if not self.instance_id:
            raise ValueError("instance_id must be nonempty")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    @property
    def node_count(self) -> int:
        return len(self.coordinates)

    @property
    def edge_count(self) -> int:
        return self.node_count * (self.node_count - 1) // 2

    @cached_property
    def coordinate_array(self) -> np.ndarray:
        return np.asarray(self.coordinates, dtype=np.float64)

    @cached_property
    def distance_matrix(self) -> np.ndarray:
        delta = self.coordinate_array[:, None, :] - self.coordinate_array[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float64))
        if not np.all(np.isfinite(distances)):
            raise RuntimeError("distance matrix contains non-finite values")
        return distances

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinates": [list(coordinate) for coordinate in self.coordinates],
            "instance_id": self.instance_id,
            "regime": self.regime,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TSPInstance:
        raw_coordinates = payload.get("coordinates")
        if not isinstance(raw_coordinates, list):
            raise ValueError("coordinates must be a JSON array")
        coordinates: list[tuple[float, float]] = []
        for index, raw in enumerate(raw_coordinates):
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError(f"coordinate {index} must be a two-element JSON array")
            coordinates.append(
                (
                    finite_float(raw[0], name=f"coordinate[{index}].x"),
                    finite_float(raw[1], name=f"coordinate[{index}].y"),
                )
            )
        return cls(
            tuple(coordinates),
            instance_id=string(payload.get("instance_id", "instance"), name="instance_id"),
            regime=string(payload.get("regime", "unspecified"), name="regime"),
            seed=integer(payload.get("seed", 0), name="seed", minimum=0),
        )


@dataclass(frozen=True, order=True, slots=True)
class CutConstraint:
    """One canonical undirected cut constraint x(delta(S)) >= 2."""

    node_count: int
    nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.node_count < 5:
            raise ValueError("cut node_count must be at least five")
        if tuple(sorted(set(self.nodes))) != self.nodes:
            raise ValueError("cut nodes must be sorted and unique")
        if 0 in self.nodes:
            raise ValueError("canonical cut side must exclude node zero")
        if not 2 <= len(self.nodes) <= self.node_count - 2:
            raise ValueError("cut must have at least two nodes on each side")
        if self.nodes and (self.nodes[0] < 0 or self.nodes[-1] >= self.node_count):
            raise ValueError("cut node index is outside the graph")

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def complement_size(self) -> int:
        return self.node_count - self.size

    @property
    def key(self) -> str:
        return ",".join(str(node) for node in self.nodes)

    def to_dict(self) -> dict[str, object]:
        return {"node_count": self.node_count, "nodes": list(self.nodes), "key": self.key}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CutConstraint:
        node_count = integer(payload.get("node_count"), name="cut.node_count", minimum=5)
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list) or not all(
            isinstance(node, int) and not isinstance(node, bool) for node in raw_nodes
        ):
            raise ValueError("cut nodes must be an integer JSON array")
        return cls(node_count=node_count, nodes=tuple(raw_nodes))


def canonical_cut(node_count: int, nodes: Iterable[int]) -> CutConstraint:
    """Canonicalize a cut by choosing the side that excludes node zero."""

    chosen = frozenset(int(node) for node in nodes)
    if any(node < 0 or node >= node_count for node in chosen):
        raise ValueError("cut contains a node outside the graph")
    if 0 in chosen:
        chosen = frozenset(range(node_count)).difference(chosen)
    return CutConstraint(node_count=node_count, nodes=tuple(sorted(chosen)))


def candidate_cut_count(node_count: int) -> int:
    """Count canonical nontrivial cut representatives without materializing them."""

    if node_count < 5:
        raise ValueError("node_count must be at least five")
    return sum(math.comb(node_count - 1, size) for size in range(2, node_count - 1))


def enumerate_candidate_cuts(node_count: int) -> tuple[CutConstraint, ...]:
    """Enumerate one representative for every nontrivial symmetric cut."""

    if node_count < 5:
        raise ValueError("node_count must be at least five")
    candidates: list[CutConstraint] = []
    nonroot = range(1, node_count)
    for size in range(2, node_count - 1):
        for nodes in itertools.combinations(nonroot, size):
            candidates.append(CutConstraint(node_count=node_count, nodes=nodes))
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class TourSolution:
    """A canonical Hamiltonian cycle and its recomputed objective."""

    order: tuple[int, ...]
    objective: float

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tour_edges(self.order)

    def to_dict(self) -> dict[str, object]:
        return {
            "order": list(self.order),
            "edges": [list(edge) for edge in self.edges],
            "objective": self.objective,
        }


@dataclass(frozen=True, slots=True)
class TourAudit:
    hamiltonian: bool
    canonical: bool
    objective_consistent: bool
    recomputed_objective: float
    maximum_degree_error: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_tour(order: Sequence[int]) -> tuple[int, ...]:
    """Return a node-zero-started, orientation-canonical cycle order."""

    if not order:
        raise ValueError("tour order must be nonempty")
    values = tuple(int(node) for node in order)
    if len(set(values)) != len(values):
        raise ValueError("tour order must not repeat nodes")
    if 0 not in values:
        raise ValueError("tour order must contain node zero")
    start = values.index(0)
    rotated = values[start:] + values[:start]
    reversed_rotated = (0,) + tuple(reversed(rotated[1:]))
    return min(rotated, reversed_rotated)


def tour_edges(order: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return sorted undirected edges of a cycle."""

    canonical = canonical_tour(order)
    edges: list[tuple[int, int]] = []
    for left, right in zip(canonical, canonical[1:] + canonical[:1], strict=True):
        edges.append((min(left, right), max(left, right)))
    return tuple(sorted(edges))


def tour_objective(instance: TSPInstance, order: Sequence[int]) -> float:
    """Recompute a tour objective from the instance distance matrix."""

    return float(sum(instance.distance_matrix[left, right] for left, right in tour_edges(order)))


def make_tour_solution(instance: TSPInstance, order: Sequence[int]) -> TourSolution:
    canonical = canonical_tour(order)
    if set(canonical) != set(range(instance.node_count)):
        raise ValueError("tour must visit every node exactly once")
    return TourSolution(order=canonical, objective=tour_objective(instance, canonical))


def audit_tour(
    instance: TSPInstance,
    solution: TourSolution,
    *,
    tolerance: float = 1e-8,
) -> TourAudit:
    order = solution.order
    hamiltonian = len(order) == instance.node_count and set(order) == set(
        range(instance.node_count)
    )
    canonical = hamiltonian and canonical_tour(order) == order
    recomputed = tour_objective(instance, order) if hamiltonian else math.inf
    objective_consistent = hamiltonian and math.isclose(
        recomputed,
        solution.objective,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )
    degrees = [0] * instance.node_count
    if hamiltonian:
        for left, right in tour_edges(order):
            degrees[left] += 1
            degrees[right] += 1
    maximum_degree_error = max((abs(degree - 2) for degree in degrees), default=2)
    return TourAudit(
        hamiltonian=hamiltonian,
        canonical=canonical,
        objective_consistent=objective_consistent,
        recomputed_objective=recomputed,
        maximum_degree_error=maximum_degree_error,
    )


def cut_value_on_tour(cut: CutConstraint, tour: TourSolution) -> int:
    """Count selected tour edges crossing a cut."""

    side = frozenset(cut.nodes)
    return sum((left in side) != (right in side) for left, right in tour.edges)


def generate_instance(
    *,
    node_count: int,
    regime: str = "uniform",
    seed: int = 0,
    instance_id: str | None = None,
) -> TSPInstance:
    """Generate a deterministic Euclidean instance under a controlled regime."""

    if node_count < 5:
        raise ValueError("node_count must be at least five")
    if regime not in SUPPORTED_REGIMES:
        raise ValueError(f"unsupported generation regime: {regime}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)

    if regime == "uniform":
        coordinates = rng.uniform(0.05, 0.95, size=(node_count, 2))
    elif regime == "clustered":
        cluster_count = 3
        centers = rng.uniform(0.18, 0.82, size=(cluster_count, 2))
        assignments = np.arange(node_count) % cluster_count
        rng.shuffle(assignments)
        coordinates = centers[assignments] + rng.normal(0.0, 0.075, size=(node_count, 2))
    elif regime == "strongly_clustered":
        centers = np.asarray(((0.20, 0.22), (0.80, 0.78)), dtype=np.float64)
        assignments = np.arange(node_count) % 2
        rng.shuffle(assignments)
        coordinates = centers[assignments] + rng.normal(0.0, 0.035, size=(node_count, 2))
    elif regime == "ring":
        angles = np.linspace(0.0, 2.0 * math.pi, node_count, endpoint=False)
        angles += rng.normal(0.0, 0.035, size=node_count)
        radii = 0.36 + rng.normal(0.0, 0.018, size=node_count)
        coordinates = np.column_stack((0.5 + radii * np.cos(angles), 0.5 + radii * np.sin(angles)))
    elif regime == "grid_jitter":
        width = int(math.ceil(math.sqrt(node_count)))
        grid = np.asarray(
            [
                ((column + 1) / (width + 1), (row + 1) / (width + 1))
                for row in range(width)
                for column in range(width)
            ][:node_count],
            dtype=np.float64,
        )
        coordinates = grid + rng.normal(0.0, 0.018, size=grid.shape)
    elif regime == "anisotropic":
        x = rng.uniform(0.05, 0.95, size=node_count)
        y = 0.5 + rng.normal(0.0, 0.075, size=node_count)
        coordinates = np.column_stack((x, y))
    elif regime == "outlier":
        core = rng.normal(loc=(0.30, 0.32), scale=(0.11, 0.10), size=(node_count - 1, 2))
        outlier = np.asarray(((0.90, 0.88),), dtype=np.float64)
        coordinates = np.vstack((core, outlier))
        rng.shuffle(coordinates)
    else:  # two_cluster_bridge
        left_count = max(2, (node_count - 2) // 2)
        right_count = node_count - 2 - left_count
        left = rng.normal(loc=(0.20, 0.50), scale=(0.045, 0.11), size=(left_count, 2))
        right = rng.normal(loc=(0.80, 0.50), scale=(0.045, 0.11), size=(right_count, 2))
        bridge = np.asarray(((0.44, 0.46), (0.56, 0.54)), dtype=np.float64)
        coordinates = np.vstack((left, bridge, right))
        rng.shuffle(coordinates)

    coordinates = np.clip(coordinates, 0.01, 0.99)
    # Break machine-precision duplicates deterministically without changing the regime materially.
    for index in range(node_count):
        for previous in range(index):
            if np.array_equal(coordinates[index], coordinates[previous]):
                coordinates[index, 0] = min(0.999999, coordinates[index, 0] + 1e-9 * (index + 1))
    identifier = instance_id or f"{regime}-n{node_count}-seed{seed}"
    return TSPInstance(
        coordinates=tuple((float(x), float(y)) for x, y in coordinates),
        instance_id=identifier,
        regime=regime,
        seed=seed,
    )


def solve_held_karp(instance: TSPInstance, *, maximum_nodes: int = 18) -> TourSolution:
    """Solve TSP exactly by Held-Karp dynamic programming for independent verification."""

    n = instance.node_count
    if n > maximum_nodes:
        raise ValueError("instance exceeds Held-Karp verification limit")
    m = n - 1
    state_count = 1 << m
    costs = np.full((state_count, m), np.inf, dtype=np.float64)
    parents = np.full((state_count, m), -1, dtype=np.int16)
    distances = instance.distance_matrix

    for terminal in range(m):
        mask = 1 << terminal
        costs[mask, terminal] = distances[0, terminal + 1]

    tolerance = 1e-12
    for mask in range(1, state_count):
        for terminal in range(m):
            terminal_bit = 1 << terminal
            if not mask & terminal_bit:
                continue
            previous_mask = mask ^ terminal_bit
            if previous_mask == 0:
                continue
            best_cost = math.inf
            best_parent = -1
            remaining = previous_mask
            while remaining:
                bit = remaining & -remaining
                predecessor = bit.bit_length() - 1
                candidate = (
                    costs[previous_mask, predecessor] + distances[predecessor + 1, terminal + 1]
                )
                if candidate < best_cost - tolerance or (
                    math.isclose(candidate, best_cost, rel_tol=0.0, abs_tol=tolerance)
                    and predecessor < best_parent
                ):
                    best_cost = float(candidate)
                    best_parent = predecessor
                remaining ^= bit
            costs[mask, terminal] = best_cost
            parents[mask, terminal] = best_parent

    full = state_count - 1
    best_total = math.inf
    best_terminal = -1
    for terminal in range(m):
        candidate = costs[full, terminal] + distances[terminal + 1, 0]
        if candidate < best_total - tolerance or (
            math.isclose(candidate, best_total, rel_tol=0.0, abs_tol=tolerance)
            and terminal < best_terminal
        ):
            best_total = float(candidate)
            best_terminal = terminal

    if best_terminal < 0 or not math.isfinite(best_total):
        raise RuntimeError("Held-Karp failed to recover an optimal tour")
    reverse_path: list[int] = []
    mask = full
    terminal = best_terminal
    while terminal >= 0:
        reverse_path.append(terminal + 1)
        parent = int(parents[mask, terminal])
        mask ^= 1 << terminal
        terminal = parent
    order = (0,) + tuple(reversed(reverse_path))
    solution = make_tour_solution(instance, order)
    if not math.isclose(solution.objective, best_total, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("Held-Karp backtracking objective is inconsistent")
    return solution


def solve_brute_force(instance: TSPInstance, *, maximum_nodes: int = 10) -> TourSolution:
    """Enumerate tours for a third independent exact check on tiny instances."""

    if instance.node_count > maximum_nodes:
        raise ValueError("instance exceeds brute-force verification limit")
    best: TourSolution | None = None
    for permutation in itertools.permutations(range(1, instance.node_count)):
        candidate = make_tour_solution(instance, (0,) + permutation)
        if (
            best is None
            or candidate.objective < best.objective - 1e-12
            or (
                math.isclose(candidate.objective, best.objective, rel_tol=0.0, abs_tol=1e-12)
                and candidate.order < best.order
            )
        ):
            best = candidate
    if best is None:
        raise RuntimeError("brute-force tour enumeration produced no solution")
    return best
