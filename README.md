# ML Warm-Start Constraint Generation

[![CI](https://github.com/jorsacademy/ml-warm-start-constraint-generation/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/ml-warm-start-constraint-generation/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first research implementation of **machine-learned warm starts for exact constraint generation**. The repository studies whether a model trained on previously solved Traveling Salesman Problem instances can predict useful subtour-elimination constraints before the first restricted MILP solve, while exact separation remains responsible for feasibility and optimality.

The learned component never invents inequality coefficients, never removes a required constraint, and never certifies convergence. It only selects an initial subset from a mathematically valid, canonical SEC universe. A poor prediction may add overhead or fail to reduce iterations; it cannot change the exact tour returned by the constraint-generation loop.

## Research question

> Under a fixed preload budget, does learning an instance-specific one-shot constraint core reduce online constraint-generation work more effectively than learning only constraints that bind at an exact tour, geometric heuristics, random preload, or cold start?

The benchmark separates four questions that are often conflated:

1. **Constraint-set prediction:** does the scorer rank offline target constraints above irrelevant valid cuts?
2. **Root formulation quality:** does the selected preload close the gap between the degree-only model and the exact tour objective?
3. **Online algorithmic work:** does warm-starting reduce master solves and newly generated constraints?
4. **End-to-end value:** after including feature construction and inference, is wall-clock performance better or worse than cold constraint generation?

No weighted composite score merges these criteria.

## Claims boundary

This is a compact synthetic methodology benchmark. It does **not** claim:

- a state-of-the-art TSP solver;
- superiority over modern branch-and-cut implementations;
- that an MLP can identify every useful SEC;
- that fewer constraint-generation iterations necessarily imply lower runtime;
- industrial-scale enumeration of the exponential SEC universe;
- exactness from the neural model;
- transfer to arbitrary MILPs without changing the representation and separator;
- reproduction of any one published paper;
- an OSI-approved open-source license.

Empirical gains are deliberately not hard-coded into the repository. The supplied protocol may show positive, neutral, or negative learned-warm-start effects depending on the regime, budget, solver overhead, and random seed.

## Optimization problem

For a complete undirected graph \(G=(V,E)\) with Euclidean edge costs \(c_e\), the symmetric TSP is modeled as

\[
\min_{x\in\{0,1\}^{|E|}}
\sum_{e\in E} c_e x_e
\]

subject to degree equalities

\[
\sum_{e\in\delta(i)}x_e=2
\qquad \forall i\in V,
\]

and subtour-elimination cut constraints

\[
\sum_{e\in\delta(S)}x_e\ge 2
\qquad
\forall S\subset V,
\quad 2\le |S|\le |V|-2.
\]

The degree-only binary master can return a disconnected two-factor: several node-disjoint cycles. For an integer solution, connected components expose violated SECs directly. The solver adds those cuts and resolves until the selected edges form one Hamiltonian cycle.

```text
valid initial SEC set
        │
        ▼
binary degree-constrained master MILP
        │
        ├── connected 2-factor → Hamiltonian tour → stop
        │
        └── disconnected cycles
                 │
                 ▼
       exact component separation
                 │
                 ▼
          add violated SECs
                 │
                 └────────────── repeat
```

Every restricted master is solved to global integer optimality with `scipy.optimize.milp` and HiGHS.

## Why a constraint warm start?

Cold constraint generation begins with no SECs. If previous problem instances reveal recurring geometric partitions, a model may preload constraints that would otherwise be discovered after one or more expensive master solves.

The central correctness principle is:

```text
learning chooses where exact optimization starts;
learning does not decide where exact optimization stops.
```

All preloaded inequalities are valid for every Hamiltonian tour. After the warm start, exact component separation continues without approximation. Therefore the method inherits the feasibility and optimality logic of the underlying constraint-generation procedure.

## Offline target sets

One exact cold run produces several distinct supervision targets. They are not treated as interchangeable.

### Initial cuts

`initial_cuts` are the component SECs found after the first degree-only master solve. They describe the first correction required by cold constraint generation, but do not necessarily suffice to solve the instance in one warm-started master call.

### Trajectory cuts

`trajectory_cuts` are all distinct SECs generated during the complete cold run. Preloading them reproduces a one-solve formulation for that exact instance, but the set may be redundant and its cardinality varies with the cold trajectory.

### Binding cuts

`binding_cuts` are canonical candidate SECs satisfying

\[
x^*(\delta(S))=2
\]

at the exact tour. Binding information is a natural but potentially misleading proxy in integer optimization: a nonbinding inequality may still be essential for excluding a better disconnected integer solution.

### Operational invariant core

The main `invariant` target is a deterministic, deletion-minimal **one-shot sufficient core** derived from the cold trajectory:

1. start with all trajectory cuts;
2. solve the restricted master and verify that it returns a connected tour with the exact objective;
3. remove one cut at a time whenever that property remains true;
4. repeat until no single remaining cut can be deleted.

The resulting set is called an operational invariant core in this repository because, for the declared instance and solver tolerance, it preserves the exact one-shot outcome. It is **not** claimed to be the unique invariant set, a globally minimum-cardinality set, or the invariant-set construction of every prior paper.

This explicit binding-versus-invariant comparison is the main methodological axis of the project.

## Canonical candidate universe

For an undirected cut, \(S\) and \(V\setminus S\) define the same inequality. The repository retains exactly one representative by requiring the stored side to exclude node `0`. Candidate sides also contain at least two nodes and leave at least two nodes outside.

The resulting candidate count is

\[
\sum_{k=2}^{n-2}{n-1\choose k}.
\]

The complete universe is enumerated for the controlled benchmark sizes, currently at most 14 nodes. This is intentionally exact and transparent, but exponential. A production extension would require candidate generation, retrieval, hierarchical screening, or solver-native cut pools rather than full enumeration.

## Constraint features

Each canonical SEC is represented by 21 inspectable, solution-independent geometric features. The features do not use the exact tour, cold trajectory, or target label at inference time.

They include:

- smaller-side fraction and size balance;
- fraction and total cost of edges crossing the cut;
- crossing-edge mean, standard deviation, minimum, and maximum relative to global distances;
- within-side distance statistics for both sides;
- crossing-to-within distance ratio;
- centroid separation and side radii;
- fractions of first-, second-, and third-nearest-neighbor arcs crossing the partition;
- graph size and global distance coefficient of variation.

Features are defined to be invariant to replacing \(S\) with its complement. Training and inference use the same versioned feature schema.

## Learned scorer

`ConstraintScorer` is a small PyTorch MLP that assigns one logit to each candidate SEC. It is deliberately simpler than a graph neural network so the benchmark isolates target-set quality and exact warm-start behavior before introducing architecture complexity.

Training uses:

- all positive constraints for the selected target;
- deterministic hard negatives selected by a geometric compactness score;
- deterministic random negatives;
- weighted binary cross-entropy for class imbalance;
- feature standardization fitted only on the training split;
- AdamW, gradient clipping, validation early stopping, and best-state restoration;
- whole-instance train/validation separation;
- schema-validated Safetensors checkpoints.

The principal comparison trains two otherwise identical models:

```text
invariant_model  → predicts the deletion-minimal one-shot core
binding_model    → predicts SECs tight at the exact tour
```

This controls model capacity while changing only the semantic target.

## Warm-start policies

The exact benchmark supports the following policies:

| Method | Initial cuts | Uses offline test labels? | Budget matched? |
| --- | --- | ---: | ---: |
| `cold` | none | no | baseline |
| `random` | uniformly sampled valid SECs | no | yes |
| `compactness` | highest geometric compactness scores | no | yes |
| `binding_model` | highest binding-model logits | no | yes |
| `invariant_model` | highest invariant-model logits | no | yes |
| `oracle_trajectory` | earliest cold-trajectory cuts | yes | yes |
| `oracle_invariant_matched` | invariant cuts capped by the common budget | yes | yes |
| `oracle_invariant_full` | complete offline one-shot core | yes | no |

The full invariant oracle is an information ceiling, not a fair deployable competitor. It is reported separately and explicitly marked as not cardinality matched.

## Exactness and reliability contract

The warm-started method remains exact because:

1. every preloaded item is a valid SEC from the canonical universe;
2. valid SECs cannot remove a Hamiltonian tour;
3. each restricted master is solved globally as a binary MILP;
4. every disconnected integer two-factor is separated by component SECs;
5. termination requires one connected two-regular spanning graph, which is a Hamiltonian cycle;
6. exact Held–Karp dynamic programming independently verifies the final objective on every benchmark instance;
7. tiny instances can additionally enumerate all tours by brute force.

The implementation fails closed on malformed cuts, duplicate constraints, nonintegral master output, degree violations, active-cut violations, disconnected solutions without a new separating cut, nonfinite model scores, inconsistent corpus labels, fingerprint mismatches, or disagreement between exact oracles.

See [`docs/exactness.md`](docs/exactness.md) for the formal exact/approximate boundary.

## Evaluation metrics

### Prediction diagnostics

- average precision over the full candidate universe;
- top-\(k\) precision and recall;
- probability of retrieving at least one positive target;
- positive and negative score summaries;
- selected-set overlap with invariant, trajectory, and binding targets.

### Formulation quality

Let \(z_0\) be the first cold degree-model objective, \(z_W\) the first warm-started master objective, and \(z^*\) the exact TSP objective. For minimization, the root gap closure is

\[
\frac{z_W-z_0}{z^*-z_0},
\]

when the denominator is nonzero. Values near one mean the initial preload makes the first master as strong as the full exact outcome.

### Online constraint-generation work

- number of master MILP solves;
- one-shot solution rate;
- number of preloaded cuts;
- number of online generated cuts;
- final active-cut count;
- aggregate master branch-and-bound nodes;
- master solve time.

### End-to-end behavior

- feature/scoring/selection time;
- exact solve time;
- total warm-start plus solve time;
- paired difference from cold start;
- deterministic paired-bootstrap intervals for master-solve reduction, online-cut reduction, and runtime difference.

A reduction in iterations is not presented as a speedup unless measured runtime also improves.

## Controlled generalization protocol

The frozen protocol trains once on moderate Euclidean instances and evaluates disjoint seeds under:

1. `interpolation` — nominal uniform geometry;
2. `size_14` — node-count extrapolation;
3. `clustered` — moderate spatial clusters;
4. `strong_clustered` — tightly separated clusters;
5. `ring` — near-circular geometry;
6. `grid_jitter` — perturbed lattice structure;
7. `anisotropic` — nearly one-dimensional geometry;
8. `outlier` — one remote node;
9. `two_cluster_bridge` — two dense regions with bridge nodes.

Scenario results remain separate. A strong in-distribution average cannot conceal a failure under topology or size shift.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.11 or 3.12 is required. CPU-only PyTorch is sufficient.

## CLI

### Generate one TSP instance

```bash
warmcg generate \
  --node-count 10 \
  --regime strongly_clustered \
  --seed 42 \
  --output artifacts/instance.json
```

### Run independent exact oracles

```bash
warmcg oracle artifacts/instance.json \
  --maximum-nodes 14 \
  --brute-force-maximum-nodes 9 \
  --output artifacts/oracle.json
```

### Build exact-labeled corpora

```bash
warmcg collect \
  --count 60 \
  --node-counts 8 10 12 \
  --regimes uniform clustered strongly_clustered \
  --seed 3200 \
  --output artifacts/train.jsonl

warmcg collect \
  --count 18 \
  --node-counts 8 10 12 \
  --regimes uniform clustered strongly_clustered \
  --seed 4200 \
  --output artifacts/validation.jsonl
```

### Train the invariant-target model

```bash
warmcg train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --target invariant \
  --epochs 40 \
  --validation-budget 8 \
  --checkpoint artifacts/invariant-scorer.safetensors \
  --output-report artifacts/invariant-training.json
```

### Train the binding-target control

```bash
warmcg train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --target binding \
  --epochs 40 \
  --validation-budget 8 \
  --checkpoint artifacts/binding-scorer.safetensors \
  --output-report artifacts/binding-training.json
```

### Solve one instance with a learned preload

```bash
warmcg solve artifacts/instance.json \
  --mode learned \
  --checkpoint artifacts/invariant-scorer.safetensors \
  --budget 8 \
  --output artifacts/solution.json
```

### Compare all warm-start policies

```bash
warmcg benchmark artifacts/test.jsonl \
  --invariant-checkpoint artifacts/invariant-scorer.safetensors \
  --binding-checkpoint artifacts/binding-scorer.safetensors \
  --scenario size_14 \
  --budget 8 \
  --bootstrap-draws 1000 \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

### Run the frozen protocol

```bash
warmcg research \
  --config configs/research_v1.json \
  --checkpoint-directory artifacts/checkpoints \
  --output-report artifacts/research-report.json
```

## Repository layout

```text
src/warmcg/
├── domain.py       # TSP instances, canonical SECs, Held–Karp and brute-force oracles
├── solver.py       # exact integer master, separation, CG loop, one-shot core reduction
├── features.py     # complement-invariant constraint geometry features
├── dataset.py      # exact labels, JSONL corpora, SHA-256 integrity checks
├── model.py        # target-specific MLP and safe Safetensors checkpoints
├── training.py     # negative sampling, class-balanced training, ranking diagnostics
├── warmstart.py    # learned, heuristic, random, cold, and oracle preload policies
├── evaluation.py   # exact benchmark, work metrics, paired bootstrap summaries
├── experiment.py   # frozen train-once/evaluate-many shift protocol
└── cli.py          # end-to-end command-line workflows
```

Additional documentation:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/exactness.md`](docs/exactness.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/research_context.md`](docs/research_context.md)
- [`docs/model_card.md`](docs/model_card.md)
- [`docs/verification.md`](docs/verification.md)

## Tests and CI

GitHub Actions runs on Python 3.11 and 3.12:

```text
package installation and dependency check
Ruff lint and formatting
strict mypy
branch-aware pytest coverage
collect → train both targets → exact oracle → benchmark smoke
```

The regression suite covers canonical SEC symmetry, candidate-count identities, deterministic generators, Held–Karp/brute-force agreement, exact constraint generation, cut validity, one-shot core deletion minimality, feature complement invariance, deterministic corpora, tamper detection, model/checkpoint validation, both training targets, all warm-start policies, exact benchmark certification, frozen configuration parsing, report serialization, and CLI workflows.

## Methodological limitations

The complete candidate SEC universe is exponential, so the current benchmark deliberately limits exact-labeled instances to at most 14 nodes. The MLP scores constraints independently after global geometric features are computed; it does not model interactions among a batch of selected cuts. Consequently, top-\(k\) constraints may be individually plausible but redundant together.

The operational invariant target is deletion-minimal with respect to the deterministic removal procedure, not globally minimum-cardinality. Alternative exact tours, solver tie-breaking, or another deletion order may yield a different sufficient set. Binding labels also depend on the selected canonical exact tour when multiple tours share the same cost.

Master solve times on these small instances are noisy, and feature enumeration may dominate solver work. Master-solve count and online-cut count are therefore primary algorithmic work measures; runtime is reported separately without hardware-independent claims.

The integer-solution separator is intentionally simple. It does not implement fractional SEC separation, comb inequalities, blossom cuts, branch-and-cut callbacks, cut aging, local cuts, or TSPLIB-scale solver engineering.

## Research context

The repository is positioned relative to:

- Jiménez-Cordero, Morales, and Pineda, [“Warm-starting constraint generation for mixed-integer optimization: A Machine Learning approach”](https://doi.org/10.1016/j.knosys.2022.109570), *Knowledge-Based Systems* 253 (2022), which motivates learning instance-dependent invariant constraint sets while retaining exact constraint-generation guarantees;
- Pferschy and Staněk, [“Generating subtour elimination constraints for the TSP from pure integer solutions”](https://arxiv.org/abs/1511.03533), which studies repeated globally solved integer degree models followed by subtour separation;
- Sambharya, Hall, Amos, and Stellato, [“End-to-End Learning to Warm-Start for Real-Time Quadratic Optimization”](https://proceedings.mlr.press/v211/sambharya23a.html), L4DC 2023, for learned warm starts as an optimization-acceleration paradigm;
- Schmidtobreick, Arnström, Häusner, and Sjölund, [“Warm-starting active-set solvers using graph neural networks”](https://proceedings.mlr.press/v331/schmidtobreick26a.html), L4DC 2026, for active-constraint prediction across parametric optimization problems;
- Dantzig, Fulkerson, and Johnson, “Solution of a Large-Scale Traveling-Salesman Problem,” *Operations Research* 2(4), 1954, for the classical cutting-plane formulation of TSP.

This implementation is not a reproduction of any one paper. Its narrower contribution is a controlled **binding-versus-one-shot-core** target comparison on an exact integer SEC-generation pipeline, with matched preload budgets and independent exact verification.

## License

PolyForm Noncommercial 1.0.0. The repository is source-available for noncommercial use; it is not offered under an OSI-approved open-source license.
