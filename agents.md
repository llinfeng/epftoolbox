# Agents Guide for epftoolbox

This file gives Codex/AI assistants the project context and working rules. Always read `.cursor/rules/main.md` first; it holds the authoritative stack, style, and compatibility constraints.

## Quick references
- Project overview and install: `README.md`
- Detailed rules and compatibility pins: `.cursor/rules/main.md`
- Notebooks folder: `Notebooks/` (ordered examples for forecasting, evaluation, datasets)
- Main package code: `epftoolbox/`; examples: `examples/`; datasets kept locally under `datasets/` (gitignored)

## Operating rules for agents
- Use `rg` for search and `apply_patch` for small edits; never reset or remove user changes.
- Follow PEP 8 and keep TensorFlow 2.x / NumPy <2 / Keras <3 compatibility; prefer adding type hints.
- Keep notebooks and docs self-contained; call helpers from `epftoolbox/` rather than duplicating logic.
- Large/local data stay in `datasets/` or `forecasts_local/`; do not add binaries to git.
- When changing APIs, update related docs/examples/notebooks to preserve reproducibility.

## Environment and setup
- Python 3.9–3.11. Recommended: `uv venv .venv && source .venv/bin/activate`.
- Install tooling and package: `uv pip install --upgrade pip`, `uv pip install "setuptools<81"`, `uv pip install "tensorflow[and-cuda]==2.15.*" -e .`.
- GPU check: `python - <<'PY'\nimport tensorflow as tf\nprint(tf.config.list_physical_devices('GPU'))\nPY`

## Notebook workflow (non-interactive)
- Execute in place: `uv run jupyter nbconvert --to notebook --execute --inplace Notebooks/<file>.ipynb`
- Parameterized runs (if `papermill` is installed): `uv run papermill Notebooks/<in>.ipynb Notebooks/<out>.ipynb -p key value`
- Keep outputs lightweight; write heavy artifacts under `forecasts_local/`.

## Validation
- Minimum check after library changes: `uv run python -c "import epftoolbox; print(epftoolbox.__version__)"`.
- Run targeted scripts/examples instead of full test sweeps when possible to save time.
