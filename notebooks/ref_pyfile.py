# %% [markdown]
# # Reference pose-model benchmark
#
# The implementation now lives in `aprilpose.benchmark`. This file remains as a
# notebook-friendly launcher so existing workflows keep working.
#
# Pipeline:
# 1. inspect and validate the reference recording
# 2. smoke-check each selected pose model
# 3. benchmark all models on the same detections and active mocap interval
# 4. write `frames.csv` and `summary.json`

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aprilpose.benchmark.cli import main


# %%
if __name__ == "__main__":
    main()
