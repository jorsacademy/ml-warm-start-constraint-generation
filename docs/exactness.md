# Exactness and Reliability Contract

## Declared problem class

The exactness claim applies to finite complete undirected Euclidean TSP instances represented by pairwise distinct two-dimensional coordinates. The solver uses binary edge variables, degree equalities, and subtour-elimination cut constraints.

## Exact components

For every benchmark instance:

- all edge costs are recomputed from coordinates;
- the restricted master is solved to global integer optimality;
- returned edge variables are audited for finiteness, integrality, bounds, degree equalities, active SECs, and objective consistency;
- disconnected integer two-factors are separated by exact component cuts;
- termination requires a connected two-regular spanning graph;
- the final tour is audited as Hamiltonian and canonical;
- Held–Karp dynamic programming independently verifies the exact objective;
- tiny instances can enumerate every tour as a third oracle;
- stored corpus labels and fingerprints are recomputed on load.

## Why arbitrary predicted subsets cannot break optimality

For every nontrivial node set \(S\), the SEC

\[
x(\delta(S))\ge 2
\]

is valid for every Hamiltonian cycle. Preloading any subset of these inequalities therefore cannot remove the true TSP optimum.

If the current binary degree-model solution is disconnected, each component is a proper cycle. Its boundary has value zero, so the corresponding SEC is violated. Adding at least one previously unseen violated component cut removes the current disconnected solution. Repeating this finite process eventually returns a connected two-factor or raises on an implementation failure.

A connected undirected graph in which every node has degree two is one Hamiltonian cycle. Because every restricted master is globally optimized and all added constraints are valid, the first connected master optimum is also optimal for the full TSP formulation.

The neural model affects only the initial valid subset. It does not affect master optimality, cut validity, separation, or the stopping rule.

## Operational invariant core

The offline core is constructed from the exact cold trajectory. The full trajectory set is first verified to return the exact connected tour in one master solve. A deterministic deletion procedure then removes any cut whose deletion preserves that property. A final pass verifies that no single remaining cut is removable.

This establishes:

- one-shot sufficiency for the declared instance;
- deletion minimality under single-cut removal;
- exact objective consistency.

It does not establish globally minimum cardinality or uniqueness.

## Approximate components

The following components are heuristic or statistical:

- target-set prediction;
- candidate scores;
- negative sampling;
- geometric compactness ranking;
- generalization across node counts or coordinate distributions;
- feature-construction and wall-clock timing;
- bootstrap confidence intervals.

These components may influence efficiency but not the final exact answer.

## Numerical scope

Distances and MILP objectives use double precision. Objective comparisons use explicit absolute and relative tolerances. Held–Karp and MILP values must agree within those tolerances. This is a numerical exactness contract for the declared benchmark, not a symbolic proof over arbitrary real input.

## Failure policy

The implementation raises rather than silently continuing when it observes:

- malformed or duplicate coordinates;
- invalid or noncanonical cuts;
- incompatible cut dimensions;
- duplicate active constraints;
- nonoptimal or malformed MILP results;
- nonintegral binary decisions;
- degree or SEC violations;
- disconnected solutions for which no new component cut is found;
- iteration-limit exhaustion;
- disagreement between exact oracles;
- inconsistent stored labels;
- corpus fingerprint mismatch;
- incompatible checkpoint metadata or tensors;
- nonfinite neural inputs, outputs, losses, or gradients.
