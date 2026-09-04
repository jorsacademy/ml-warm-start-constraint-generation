# Research Context

## Constraint generation with learned initialization

Constraint generation is exact but can require repeated optimization of restricted models. Jiménez-Cordero, Morales, and Pineda formalized a machine-learning-assisted warm-start strategy based on instance-specific invariant constraint sets and emphasized retaining the feasibility and optimality guarantees of the underlying exact procedure.

This repository adopts that separation of responsibilities: learning proposes an initial valid subset, while exact constraint generation remains active until its mathematical stopping condition is met.

Reference:

- A. Jiménez-Cordero, J. M. Morales, and S. Pineda, “Warm-starting constraint generation for mixed-integer optimization: A Machine Learning approach,” *Knowledge-Based Systems* 253, 109570, 2022. https://doi.org/10.1016/j.knosys.2022.109570

## Pure-integer TSP separation

Modern exact TSP solvers use sophisticated branch-and-cut machinery, often separating inequalities at fractional solutions. Pferschy and Staněk studied a deliberately simpler pure-integer loop: solve the degree model to integer optimality, detect subtours, add SECs, and repeat. That structure is particularly suitable for a verification-first learning benchmark because violated constraints are unambiguous connected components.

Reference:

- U. Pferschy and R. Staněk, “Generating subtour elimination constraints for the TSP from pure integer solutions,” arXiv:1511.03533, 2015. https://arxiv.org/abs/1511.03533

## Learned warm starts more broadly

Learning to warm-start appears in several optimization settings. Sambharya et al. learn initial iterates for first-order quadratic optimization, while Schmidtobreick et al. predict active constraints for an active-set QP solver. These works differ in algorithm and problem class but share a central design question: can instance structure predict a better solver starting state without replacing the solver’s correctness machinery?

References:

- R. Sambharya, G. Hall, B. Amos, and B. Stellato, “End-to-End Learning to Warm-Start for Real-Time Quadratic Optimization,” PMLR 211, 2023. https://proceedings.mlr.press/v211/sambharya23a.html
- E. J. Schmidtobreick, D. Arnström, P. Häusner, and J. Sjölund, “Warm-starting active-set solvers using graph neural networks,” PMLR 331, 2026. https://proceedings.mlr.press/v331/schmidtobreick26a.html

## Distinction from learned cut selection

A separate line of work learns which currently violated cuts to add during branch-and-cut. This repository instead predicts a constraint set **before** the first online solve and then uses a deterministic exact separator. It therefore studies formulation initialization rather than per-iteration cut ranking.

The repository also differs from learned cut generation. It never predicts coefficients or validity. Every candidate is a canonical subtour-elimination constraint whose validity is known analytically.

## Specific methodological contribution

The benchmark isolates a question that is especially important in integer optimization:

> Is learning constraints that bind at one exact solution sufficient, or is it more effective to learn nonbinding constraints that are necessary to recover the exact optimum from a reduced formulation?

It operationalizes the second target as a deletion-minimal one-shot core extracted from the exact cold trajectory and compares both targets under the same model capacity and preload budget.
