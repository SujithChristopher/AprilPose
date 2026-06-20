## Reference model benchmark

The reference benchmark is a staged pipeline that validates the recording,
smoke-checks each pose model, benchmarks all selected models on shared
detections, and writes per-frame and aggregate results.

```bash
uv run aprilpose-benchmark --max-active-frames 300
```

Outputs are written to `outputs/reference_benchmark/`:

- `frames.csv` contains per-frame poses, runtime, and refinement diagnostics.
- `summary.json` contains dataset inspection, model checks, translation and
  SO(3) orientation-change metrics, and timing summaries.

Available models are `baseline`, `subpix_ippe_temporal`, and
`internal_lines`. Select a subset with:

```bash
uv run aprilpose-benchmark --models baseline internal_lines
```

`notebooks/ref_pyfile.py` remains as a compatibility launcher. The reusable
implementation lives under `aprilpose/benchmark/`.
