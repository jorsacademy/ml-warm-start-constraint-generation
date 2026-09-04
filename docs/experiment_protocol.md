# Experiment Protocol

## Primary comparison

The experiment compares an invariant-core target with a binding-constraint target using the same:

- candidate universe;
- 21-dimensional feature schema;
- MLP architecture;
- training and validation instances;
- optimizer and class-balancing strategy;
- warm-start cardinality budget;
- exact downstream solver;
- evaluation seeds.

Only the supervision target differs.

## Data splits

Splits are made by complete TSP instance. Constraint rows from one instance never appear in multiple splits. This prevents thousands of highly correlated candidate SECs from the same graph being divided between training and validation.

Training, validation, and every evaluation scenario use disjoint deterministic seed ranges. Dataset fingerprints are included in training and evaluation reports.

## Label construction

For each training instance:

1. run cold integer SEC generation to completion;
2. verify the final objective with Held–Karp;
3. record first-iteration and full-trajectory cuts;
4. greedily reduce the trajectory to a deletion-minimal one-shot core;
5. enumerate canonical SEC candidates;
6. mark constraints tight at the exact tour as binding labels.

All offline labeling work is excluded from online warm-start runtime.

## Training sample construction

Positive labels are rare relative to the exponential candidate universe. Every positive candidate is retained. Negatives combine:

- high-scoring geometric hard negatives;
- deterministic random negatives;
- a minimum number per instance.

Feature normalization is fitted only on training samples. Validation uses the frozen normalization and the full candidate universe for ranking metrics.

## Fixed-budget policies

`random`, `compactness`, `binding_model`, `invariant_model`, `oracle_trajectory`, and `oracle_invariant_matched` receive the same cardinality cap. `cold` receives no preload. `oracle_invariant_full` is explicitly excluded from budget-matched comparisons.

If a target oracle contains fewer than the requested budget, it returns all available target cuts rather than padding with unrelated inequalities.

## Primary metrics

Primary algorithmic work metrics:

- master-solve count;
- one-shot rate;
- number of online generated cuts;
- first-master root gap closure.

Secondary metrics:

- aggregate master MIP nodes;
- solve time;
- selection time;
- total runtime;
- invariant/trajectory/binding overlap.

All final solutions must be certified by the exact loop and verified by Held–Karp.

## Paired uncertainty estimates

For each method and instance, differences from cold start are computed before bootstrapping. Deterministic paired bootstrap intervals are reported for:

- master-solve reduction;
- online-cut reduction;
- total-runtime difference.

Intervals quantify finite-sample variation; they do not establish performance on a population beyond the synthetic generator.

## Frozen scenarios

The checked-in `configs/research_v1.json` trains on uniform and clustered instances with node counts 8, 10, and 12. It evaluates separately on:

- nominal interpolation;
- 14-node size shift;
- clustered and strongly clustered geometry;
- ring geometry;
- jittered grid geometry;
- anisotropic geometry;
- an isolated outlier;
- two clusters connected by bridge nodes.

No evaluation scenario is used for model selection.

## Interpretation rules

A learned method should not be described as better merely because its target-classification score is higher. The downstream exact work metrics are decisive.

A reduction in master solves should not be described as a speedup if feature enumeration and inference make total runtime worse.

A full-oracle result should not be compared as though it respected the common preload budget.

A one-shot rate below one is not a correctness failure. Exact online separation is expected to repair an incomplete warm start.
