# Architecture

## Separation of concerns

The package separates mathematical validity, offline supervision, statistical learning, online selection, and exact optimization.

```text
Offline exact labeling
  TSP generator
      → cold integer SEC generation
      → exact Held–Karp verification
      → trajectory / first / binding labels
      → deletion-minimal one-shot core
      → versioned JSONL corpus

Statistical learning
  candidate SEC geometry
      → deterministic negative sampling
      → target-specific MLP
      → validation ranking
      → Safetensors checkpoint

Online exact solution
  unseen TSP instance
      → enumerate canonical valid SEC candidates
      → rank and preload at most k constraints
      → exact binary master
      → exact component separation until connected
      → independent Held–Karp objective audit
```

## Modules

`domain.py` owns immutable TSP data, canonical cut representations, exact tour audits, deterministic synthetic regimes, Held–Karp dynamic programming, and tiny brute-force verification.

`solver.py` owns the binary degree-constrained master, component separation, exact constraint generation, and deletion-minimal one-shot-core construction. No neural code appears in this layer.

`features.py` converts a valid SEC candidate into a fixed, versioned geometry vector. Features are complement-invariant and do not inspect exact labels.

`dataset.py` runs offline exact optimization, constructs all target sets, serializes whole-instance records, computes stable fingerprints, and recomputes exact labels when loading a corpus.

`model.py` defines the target-specific MLP, feature normalization, batched inference, and schema-validated Safetensors checkpoints.

`training.py` handles deterministic negative sampling, class weighting, early stopping, and full-universe ranking evaluation.

`warmstart.py` implements all deployable and oracle preload policies. It can only return `CutConstraint` values already validated by the domain layer.

`evaluation.py` runs every policy through the same exact solver and reports prediction, root-strength, online-work, exactness, and runtime metrics separately.

`experiment.py` fixes train/validation seeds, trains the invariant and binding controls once, then evaluates them under disjoint geometry and size shifts.

## Trust boundaries

The neural checkpoint is treated as untrusted input. Loading validates the checkpoint schema, feature schema, target semantics, model dimensions, tensor keys, and finite parameter values. Model outputs are checked for shape and finiteness before ranking.

Corpus files are also treated as untrusted input. Loading recomputes the SHA-256 fingerprint, exact tour objective, trajectory sufficiency, invariant-core sufficiency, binding tightness, and candidate count.

The exact solver never receives arbitrary user-generated inequality coefficients. Initial cuts must be instances of the canonical `CutConstraint` type, from which the SEC row is constructed internally.

## Extensibility

A larger-scale extension can replace full candidate enumeration with a candidate generator while preserving the same interface:

```python
select_warm_start(instance, method=..., budget=...)
```

Likewise, a GNN or set-scoring architecture can replace the MLP as long as it emits finite scores for canonical candidates. Exactness remains in `solver.py`, not in the learned component.
