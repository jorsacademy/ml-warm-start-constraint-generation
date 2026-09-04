# Security Policy

## Supported version

The current `main` branch is the supported research version.

## Reporting

Report security issues privately to the repository owner rather than opening a public exploit report.

## Threat model

Corpus JSONL files, instance JSON, research configurations, and model checkpoints should be treated as untrusted inputs.

The implementation mitigates common risks by:

- validating all JSON scalar and array types;
- recomputing corpus fingerprints and exact labels;
- using Safetensors rather than pickle for checkpoints;
- loading state dictionaries strictly;
- bounding supported exact-label instance sizes;
- rejecting nonfinite numerical data and neural outputs;
- constructing SEC coefficient rows internally from typed node subsets;
- avoiding shell execution, dynamic Python evaluation, and arbitrary code generation.

The project is research software and has not undergone an external security audit. Run untrusted workloads in an isolated environment with resource limits.
