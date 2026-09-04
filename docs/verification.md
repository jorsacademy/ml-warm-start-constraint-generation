# Verification

The pre-merge local verification baseline exercises the exact and learned pipeline:

- 38 regression tests pass;
- branch-aware coverage is above the configured 84% threshold;
- Python compile-all succeeds for source and tests;
- Held–Karp agrees with brute-force tour enumeration on tiny instances;
- cold and warm-started constraint generation return the Held–Karp optimum;
- generated SECs are valid and incompatible cuts are rejected;
- the offline one-shot core is verified and deletion-minimal under single-cut removal;
- complement-equivalent cuts produce identical features;
- corpus fingerprints and exact labels are recomputed on load;
- both invariant and binding training paths produce finite models;
- Safetensors checkpoints round-trip with schema validation;
- every benchmark method remains exact after online completion;
- research-configuration parsing, report serialization, and CLI workflows complete locally.

GitHub Actions is the authoritative clean-environment check. It runs Ruff linting and formatting, strict mypy, branch-aware pytest, and an end-to-end collect–train-both-targets–oracle–benchmark smoke workflow on Python 3.11 and 3.12.
