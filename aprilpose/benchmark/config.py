from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_RECORDING = Path("data/ref_recording_single_april_diwakar")
DEFAULT_MODELS = ("baseline", "subpix_ippe_temporal", "internal_lines")


@dataclass(frozen=True)
class BenchmarkConfig:
    recording_dir: Path = DEFAULT_RECORDING
    calibration_path: Path = Path("calibration/diwakar_calibration.toml")
    output_dir: Path = Path("outputs/reference_benchmark")
    tag_id: int = 12
    max_active_frames: int = 300
    model_names: tuple[str, ...] = DEFAULT_MODELS
    samples_per_segment: int = 5
    profile_radius: int = 5
    min_gradient: float = 8.0
    min_line_points: int = 5
    max_line_rms: float = 2.5

    @property
    def frames_path(self) -> Path:
        return self.recording_dir / "webcam_color.msgpack"

    @property
    def timestamps_path(self) -> Path:
        return self.recording_dir / "webcam_timestamp.msgpack"

    @property
    def mocap_path(self) -> Path:
        return self.recording_dir / f"{self.recording_dir.name}.csv"

