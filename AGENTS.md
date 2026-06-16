# Repository Guidelines

## Project Structure & Module Organization

This is a new Python project for AprilTag pose refinement experiments.

- `main.py` is the current executable entry point.
- `docs/` contains design notes, including `docs/internal-line-pose-refinement.md`.
- `pyproject.toml` defines the package metadata, Python version, and dependencies.
- `uv.lock` pins the dependency graph for reproducible local environments.

As the project grows, place reusable code under an `aprilpose/` package and keep
scripts or demos separate from library code. Put tests under `tests/` with paths
that mirror the package structure.

## Build, Test, and Development Commands

Use `uv` for environment and dependency management:

```bash
uv sync
uv run python main.py
```

- `uv sync` creates or updates the local environment from `pyproject.toml` and
  `uv.lock`.
- `uv run python main.py` runs the current entry point.

When tests are added, prefer:

```bash
uv run pytest
```

## Coding Style & Naming Conventions

Use Python 3.13+ syntax. Follow PEP 8 with 4-space indentation, clear function
names, and small modules. Prefer explicit names such as `edge_segments`,
`camera_matrix`, and `refine_pose_from_lines` over abbreviations.

Use `snake_case` for functions, variables, and modules. Use `PascalCase` for
classes and dataclasses. Keep numerical computer-vision code typed where useful,
especially for array shapes, camera parameters, and return values.

## Testing Guidelines

No test suite exists yet. Add tests under `tests/` as implementation begins.
Name test files `test_*.py` and test functions `test_*`.

Prioritize deterministic unit tests for geometry helpers first:

- tag-local to metric coordinate conversion
- expected edge-segment generation
- line intersection math
- pose-refinement input validation

Use synthetic image fixtures sparingly and keep large generated artifacts out of
git unless they are small and essential.

## Commit & Pull Request Guidelines

The repository currently has only an initial commit, so no detailed commit
convention is established. Use concise imperative commit messages, for example:

```text
Add internal line refinement design doc
Implement edge segment generation
```

Pull requests should include a short summary, test results, and any relevant
before/after pose or timing measurements. For image-processing changes, include
sample inputs or visual diagnostics when they materially clarify the behavior.

## Agent-Specific Instructions

Shell commands in this environment should be run through `rtk`, for example:

```bash
rtk uv run python main.py
```

Do not overwrite existing contributor instructions. If `AGENTS.md` already
exists in a future checkout, read it and preserve its guidance.
