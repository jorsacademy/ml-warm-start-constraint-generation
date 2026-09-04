# Contributing

Changes should preserve the repository’s exactness boundary: learned components may rank valid constraints but must not determine validity, remove exact separation, or certify convergence.

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

New mathematical behavior requires tests. In particular:

- a new cut family needs an explicit validity argument and independent audit;
- a new exact oracle should be compared against an existing independent oracle on overlapping sizes;
- a new feature must be finite, deterministic, documented, and versioned;
- a new checkpoint format must reject incompatible schemas;
- a new benchmark method must use the same exact completion loop;
- a change to target construction must update the claims boundary and corpus schema.

Do not commit generated datasets, model checkpoints, benchmark artifacts, cache directories, API credentials, or proprietary solver files.
