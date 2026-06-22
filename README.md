# QLBS Benchmark

Reproducibility repository accompanying the FISAT 2026 QLBS study. It contains
controller implementations, the benchmark environment, configurations and utility
matrices, experiment scripts, generated CSV results, and figure-generation code.

The repository does **not** include the manuscript, reviewer correspondence, or
obsolete drafts. Values transcribed from the manuscript are kept separately in
`results/paper_reference/`; fresh outputs produced by this code are written to
`results/generated/`. This prevents archived paper values from being mistaken for
newly reproduced measurements.

## Repository layout

```text
src/                    controller implementations
benchmark/              environment, simulator, metrics, and controller factory
configs/                 experiment settings and QLBS utility matrices
scripts/                 experiment entry points
results/paper_reference/ manuscript values retained for audit
results/generated/       outputs created by the current implementation
figures/                 generated PDF and PNG plots
logs/                    per-tick logs (generated locally; not committed)
```

Implemented controllers are QLBS, Utility AI, Softmax Utility AI,
MarkovBelief, FSM, deterministic Behavior Tree, and Stochastic Behavior Tree.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce everything

The full pipeline runs the benchmark, independent-seed n-gram evaluation, QLBS
variant sweep, context-persistence analysis, learned adversary, and figures:

```bash
python reproduce_all.py
```

For a short installation and integration check:

```bash
python reproduce_all.py --quick
```

`--quick` is a smoke test. Its numerical outputs must not be reported as paper
results.

## Reproduce individual artifacts

```bash
python scripts/run_benchmark.py
python scripts/run_ngram_multiseed.py
python scripts/run_qlbs_variants.py
python scripts/run_mixing_time.py
python scripts/qlbs_learned_adversary.py
python scripts/generate_figures.py
```

The learned adversary above is a dependency-light NumPy MLP. An optional GRU
evaluation is available for machines with PyTorch:

```bash
python -m pip install -r requirements-gru.txt
python scripts/export_learned_logs.py
python scripts/qlbs_gru_adversary.py
```

## Configuration and outputs

- `configs/benchmark.yaml` is the canonical benchmark configuration.
- `configs/matrices/U_*.csv` stores every QLBS utility matrix explicitly.
- `configs/quick.yaml` only overrides workload size for smoke tests.
- Scripts use fixed, documented seeds and separate random generators for player
  behavior and controller decisions.
- Generated tables are CSV files in `results/generated/`; plots are produced in
  `figures/`. A full run overwrites any quick-test outputs.

## Scope and provenance note

The original historical multi-controller experiment harness was not present in
the supplied supporting materials. This benchmark is a clean, inspectable
implementation based on the manuscript configuration and supplied QLBS tooling.
Accordingly, exact equality with archived manuscript values should be verified,
not assumed. This limitation is documented instead of silently presenting a
reimplementation as the original execution environment.

## License

Code is released under the MIT License. See `LICENSE`.
