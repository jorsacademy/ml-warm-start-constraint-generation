"""Exact integer constraint generation for TSP subtour elimination constraints."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from warmcg.domain import (
    CutConstraint,
    TSPInstance,
    TourSolution,
    canonical_cut,
    make_tour_solution,
    solve_held_karp,
)


@dataclass(frozen=True, slots=True)
class EdgeIndex:
    """Stable complete-graph edge indexing."""

    node_count: int
    edges: tuple[tuple[int, int], ...]
    index_by_edge: dict[tuple[int, int], int]

    @classmethod
    def build(cls, node_count: int) -> EdgeIndex:
        edges = tuple(
            (left, right)
            for left in range(node_count)
            for right in range(left + 1, node_count)
        )
        return cls(
            node_count=node_count,
            edges=edges,
            index_by_edge={edge: index for index, edge in enumerate(edges)},
        )

    def index(self, left: int, right: int) -> int:
        edge = (min(left, right), max(left, right))
        try:
            return self.index_by_edge[edge]
        except KeyError as exc:
            raise ValueError("edge is outside the complete graph") from exc

    def cut_indices(self, cut: CutConstraint) -> tuple[int, ...]:
        if cut.node_count != self.node_count:
            raise ValueError("cut and edge index node counts differ")
        side = frozenset(cut.nodes)
        return tuple(
            index
            for index, (left, right) in enumerate(self.edges)
            if (left in side) != (right in side)
        )


@dataclass(frozen=True, slots=True)
class MasterSolution:
    """One globally optimal degree-plus-SEC integer master solution."""

    objective: float
    selected_edges: tuple[tuple[int, int], ...]
    components: tuple[tuple[int, ...], ...]
    tour: TourSolution | None
    solve_seconds: float
    mip_node_count: int
    mip_gap: float
    mip_dual_bound: float
    active_cut_count: int

    @property
    def connected(self) -> bool:
        return len(self.components) == 1

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["selected_edges"] = [list(edge) for edge in self.selected_edges]
        payload["components"] = [list(component) for component in self.components]
        payload["tour"] = None if self.tour is None else self.tour.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ConstraintGenerationIteration:
    iteration: int
    active_cut_count_before_solve: int
    master_objective: float
    component_count: int
    components: tuple[tuple[int, ...], ...]
    generated_cuts: tuple[CutConstraint, ...]
    solve_seconds: float
    mip_node_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "active_cut_count_before_solve": self.active_cut_count_before_solve,
            "master_objective": self.master_objective,
            "component_count": self.component_count,
            "components": [list(component) for component in self.components],
            "generated_cuts": [cut.to_dict() for cut in self.generated_cuts],
            "solve_seconds": self.solve_seconds,
            "mip_node_count": self.mip_node_count,
        }


@dataclass(frozen=True, slots=True)
class ConstraintGenerationResult:
    """Certified result of warm-started exact constraint generation."""

    initial_cuts: tuple[CutConstraint, ...]
    generated_cuts: tuple[CutConstraint, ...]
    active_cuts: tuple[CutConstraint, ...]
    iterations: tuple[ConstraintGenerationIteration, ...]
    solution: TourSolution
    total_solve_seconds: float
    total_runtime_seconds: float
    total_mip_nodes: int
    certified: bool
    held_karp_verified: bool

    @property
    def master_solve_count(self) -> int:
        return len(self.iterations)

    @property
    def online_generated_cut_count(self) -> int:
        return len(self.generated_cuts)

    @property
    def one_shot(self) -> bool:
        return self.master_solve_count == 1

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_cuts": [cut.to_dict() for cut in self.initial_cuts],
            "generated_cuts": [cut.to_dict() for cut in self.generated_cuts],
            "active_cuts": [cut.to_dict() for cut in self.active_cuts],
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "solution": self.solution.to_dict(),
            "total_solve_seconds": self.total_solve_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "total_mip_nodes": self.total_mip_nodes,
            "master_solve_count": self.master_solve_count,
            "online_generated_cut_count": self.online_generated_cut_count,
            "one_shot": self.one_shot,
            "certified": self.certified,
            "held_karp_verified": self.held_karp_verified,
        }


@dataclass(frozen=True, slots=True)
class OneShotCoreResult:
    """A deterministic inclusion-minimal trajectory subset under the declared master solver."""

    cuts: tuple[CutConstraint, ...]
    master_solve_count: int
    removed_cut_count: int
    verified_one_shot: bool
    deletion_minimal: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "cuts": [cut.to_dict() for cut in self.cuts],
            "master_solve_count": self.master_solve_count,
            "removed_cut_count": self.removed_cut_count,
            "verified_one_shot": self.verified_one_shot,
            "deletion_minimal": self.deletion_minimal,
        }


def _unique_cuts(
    instance: TSPInstance,
    cuts: Iterable[CutConstraint],
) -> tuple[CutConstraint, ...]:
    by_nodes: dict[tuple[int, ...], CutConstraint] = {}
    for cut in cuts:
        if cut.node_count != instance.node_count:
            raise ValueError("cut node count does not match instance")
        by_nodes[cut.nodes] = cut
    return tuple(sorted(by_nodes.values()))


def _connected_components(
    node_count: int,
    edges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    unseen = set(range(node_count))
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda values: (len(values), values)))


def _tour_from_edges(
    instance: TSPInstance,
    edges: Sequence[tuple[int, int]],
) -> TourSolution:
    adjacency: list[list[int]] = [[] for _ in range(instance.node_count)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    if any(len(neighbors) != 2 for neighbors in adjacency):
        raise RuntimeError("connected master solution is not two-regular")
    order = [0]
    previous = -1
    current = 0
    while True:
        options = sorted(neighbor for neighbor in adjacency[current] if neighbor != previous)
        if not options:
            raise RuntimeError("tour traversal reached a dead end")
        next_node = options[0]
        if next_node == 0:
            if len(order) != instance.node_count:
                # The smaller neighbor may close only after all nodes are visited.
                if len(options) == 1:
                    raise RuntimeError("tour closed before visiting every node")
                next_node = options[1]
            else:
                break
        if next_node in order:
            raise RuntimeError("tour traversal repeated a non-root node")
        order.append(next_node)
        previous, current = current, next_node
        if len(order) > instance.node_count:
            raise RuntimeError("tour traversal exceeded the node count")
    return make_tour_solution(instance, order)


def _result_float(result: object, name: str, default: float) -> float:
    value = getattr(result, name, default)
    if value is None:
        return default
    converted = float(value)
    return converted if math.isfinite(converted) else default


def _result_int(result: object, name: str, default: int = 0) -> int:
    value = getattr(result, name, default)
    if value is None:
        return default
    converted = int(value)
    return max(0, converted)


def solve_master(
    instance: TSPInstance,
    cuts: Iterable[CutConstraint] = (),
    *,
    time_limit_seconds: float | None = None,
) -> MasterSolution:
    """Solve the degree-constrained TSP master with the supplied valid cut constraints."""

    active_cuts = _unique_cuts(instance, cuts)
    edge_index = EdgeIndex.build(instance.node_count)
    objective = np.asarray(
        [instance.distance_matrix[left, right] for left, right in edge_index.edges],
        dtype=np.float64,
    )
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for node in range(instance.node_count):
        row = len(lower_bounds)
        for edge_position, (left, right) in enumerate(edge_index.edges):
            if left == node or right == node:
                row_indices.append(row)
                column_indices.append(edge_position)
                values.append(1.0)
        lower_bounds.append(2.0)
        upper_bounds.append(2.0)

    for cut in active_cuts:
        row = len(lower_bounds)
        indices = edge_index.cut_indices(cut)
        if len(indices) < 4:
            raise RuntimeError("a nontrivial complete-graph cut must contain at least four edges")
        for edge_position in indices:
            row_indices.append(row)
            column_indices.append(edge_position)
            values.append(1.0)
        lower_bounds.append(2.0)
        upper_bounds.append(math.inf)

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(lower_bounds), len(edge_index.edges)),
        dtype=np.float64,
    ).tocsr()
    options: dict[str, object] = {"disp": False, "presolve": True}
    if time_limit_seconds is not None:
        if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be finite and positive")
        options["time_limit"] = time_limit_seconds

    start = time.perf_counter()
    result = milp(
        c=objective,
        integrality=np.ones(len(edge_index.edges), dtype=np.int8),
        bounds=Bounds(np.zeros(len(edge_index.edges)), np.ones(len(edge_index.edges))),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        options=options,
    )
    solve_seconds = time.perf_counter() - start
    if not bool(result.success) or int(result.status) != 0 or result.x is None:
        raise RuntimeError(f"master MILP did not solve to global optimality: {result.message}")
    raw = np.asarray(result.x, dtype=np.float64)
    if raw.shape != (len(edge_index.edges),) or not np.all(np.isfinite(raw)):
        raise RuntimeError("master returned a malformed decision vector")
    integrality_violation = float(np.max(np.abs(raw - np.rint(raw))))
    if integrality_violation > 1e-6:
        raise RuntimeError("master returned a nonintegral incumbent despite binary variables")
    selected_indices = tuple(int(index) for index in np.flatnonzero(raw >= 0.5))
    selected_edges = tuple(edge_index.edges[index] for index in selected_indices)
    if len(selected_edges) != instance.node_count:
        raise RuntimeError("a two-regular spanning solution must select exactly n edges")
    degrees = [0] * instance.node_count
    for left, right in selected_edges:
        degrees[left] += 1
        degrees[right] += 1
    if any(degree != 2 for degree in degrees):
        raise RuntimeError("master solution violates a degree equality")
    for cut in active_cuts:
        side = frozenset(cut.nodes)
        cut_value = sum((left in side) != (right in side) for left, right in selected_edges)
        if cut_value < 2:
            raise RuntimeError("master solution violates an active subtour constraint")

    components = _connected_components(instance.node_count, selected_edges)
    tour = _tour_from_edges(instance, selected_edges) if len(components) == 1 else None
    recomputed_objective = float(sum(objective[index] for index in selected_indices))
    if not math.isclose(recomputed_objective, float(result.fun), rel_tol=1e-8, abs_tol=1e-8):
        raise RuntimeError("master objective is inconsistent with selected edges")
    if tour is not None and not math.isclose(
        tour.objective,
        recomputed_objective,
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise RuntimeError("tour objective is inconsistent with master edges")

    return MasterSolution(
        objective=recomputed_objective,
        selected_edges=selected_edges,
        components=components,
        tour=tour,
        solve_seconds=solve_seconds,
        mip_node_count=_result_int(result, "mip_node_count"),
        mip_gap=_result_float(result, "mip_gap", 0.0),
        mip_dual_bound=_result_float(result, "mip_dual_bound", recomputed_objective),
        active_cut_count=len(active_cuts),
    )


def violated_component_cuts(
    instance: TSPInstance,
    solution: MasterSolution,
) -> tuple[CutConstraint, ...]:
    """Return all unique SECs violated by a disconnected integer two-factor."""

    if solution.connected:
        return ()
    cuts: dict[tuple[int, ...], CutConstraint] = {}
    for component in solution.components:
        cut = canonical_cut(instance.node_count, component)
        cuts[cut.nodes] = cut
    return tuple(sorted(cuts.values()))


def run_constraint_generation(
    instance: TSPInstance,
    *,
    initial_cuts: Iterable[CutConstraint] = (),
    add_all_violated: bool = True,
    maximum_iterations: int = 100,
    verify_with_held_karp: bool = True,
    held_karp_maximum_nodes: int = 14,
) -> ConstraintGenerationResult:
    """Run exact integer SEC generation from a learned or handcrafted initial cut set."""

    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    initial = _unique_cuts(instance, initial_cuts)
    active = list(initial)
    active_keys = {cut.nodes for cut in active}
    generated: list[CutConstraint] = []
    history: list[ConstraintGenerationIteration] = []
    start = time.perf_counter()

    final_master: MasterSolution | None = None
    for iteration_index in range(1, maximum_iterations + 1):
        master = solve_master(instance, active)
        candidate_cuts = violated_component_cuts(instance, master)
        new_cuts = [cut for cut in candidate_cuts if cut.nodes not in active_keys]
        if not add_all_violated and new_cuts:
            new_cuts = [min(new_cuts, key=lambda cut: (cut.size, cut.nodes))]
        history.append(
            ConstraintGenerationIteration(
                iteration=iteration_index,
                active_cut_count_before_solve=len(active),
                master_objective=master.objective,
                component_count=len(master.components),
                components=master.components,
                generated_cuts=tuple(new_cuts),
                solve_seconds=master.solve_seconds,
                mip_node_count=master.mip_node_count,
            )
        )
        if master.connected:
            final_master = master
            break
        if not new_cuts:
            raise RuntimeError("separator found no new cut for a disconnected master solution")
        for cut in new_cuts:
            active.append(cut)
            active_keys.add(cut.nodes)
            generated.append(cut)
    if final_master is None or final_master.tour is None:
        raise RuntimeError("constraint generation exceeded its iteration limit")

    held_karp_verified = False
    if verify_with_held_karp and instance.node_count <= held_karp_maximum_nodes:
        oracle = solve_held_karp(instance, maximum_nodes=held_karp_maximum_nodes)
        if not math.isclose(
            final_master.tour.objective,
            oracle.objective,
            rel_tol=1e-8,
            abs_tol=1e-8,
        ):
            raise RuntimeError("constraint generation disagrees with Held-Karp")
        held_karp_verified = True

    return ConstraintGenerationResult(
        initial_cuts=initial,
        generated_cuts=tuple(generated),
        active_cuts=tuple(active),
        iterations=tuple(history),
        solution=final_master.tour,
        total_solve_seconds=sum(item.solve_seconds for item in history),
        total_runtime_seconds=time.perf_counter() - start,
        total_mip_nodes=sum(item.mip_node_count for item in history),
        certified=True,
        held_karp_verified=held_karp_verified,
    )


def build_one_shot_core(
    instance: TSPInstance,
    trajectory_cuts: Sequence[CutConstraint],
    *,
    optimum_objective: float,
    tolerance: float = 1e-8,
) -> OneShotCoreResult:
    """Greedily remove trajectory cuts while preserving a one-solve optimal tour outcome."""

    active = list(_unique_cuts(instance, trajectory_cuts))
    solve_count = 0

    def sufficient(cuts: Sequence[CutConstraint]) -> bool:
        nonlocal solve_count
        result = solve_master(instance, cuts)
        solve_count += 1
        return result.connected and math.isclose(
            result.objective,
            optimum_objective,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )

    if not sufficient(active):
        raise RuntimeError(
            "the full trajectory cut set does not recover the exact tour in one solve"
        )

    changed = True
    while changed:
        changed = False
        for cut in tuple(reversed(active)):
            trial = [candidate for candidate in active if candidate != cut]
            if sufficient(trial):
                active = trial
                changed = True

    deletion_minimal = True
    for cut in active:
        if sufficient([candidate for candidate in active if candidate != cut]):
            deletion_minimal = False
            break
    verified = sufficient(active)
    return OneShotCoreResult(
        cuts=tuple(active),
        master_solve_count=solve_count,
        removed_cut_count=len(_unique_cuts(instance, trajectory_cuts)) - len(active),
        verified_one_shot=verified,
        deletion_minimal=deletion_minimal,
    )
