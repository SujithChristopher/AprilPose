from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import msgpack
import msgpack_numpy
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.line_refine_single_frame import Calibration, load_calibration
from scripts.pd_support import get_rb_marker_name, read_rigid_body_csv

from .config import BenchmarkConfig


def iter_msgpack(path: Path) -> Iterator[Any]:
    with path.open("rb") as file:
        yield from msgpack.Unpacker(
            file,
            object_hook=msgpack_numpy.decode,
            raw=False,
        )


def parse_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(np.datetime64(value, "ns").astype(np.int64)) / 1e9
    if isinstance(value, (list, tuple)) and value:
        return parse_timestamp(value[-1])
    return None


def read_timestamps(path: Path) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    for item in iter_msgpack(path):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        timestamp = parse_timestamp(item[1])
        if timestamp is not None:
            result.append((int(item[0]), timestamp))
    return result


@dataclass(frozen=True)
class ReferenceInspection:
    frame_count: int
    timestamp_count: int
    active_frame_count: int
    frame_shape: tuple[int, int]
    frame_dtype: str
    mocap_sample_count: int
    mocap_duration_s: float
    calibration_method: str
    marker_length_m: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MocapTrajectory:
    times_s: np.ndarray
    positions_m: np.ndarray
    rotations: Rotation


def read_mocap(path: Path) -> MocapTrajectory:
    dataframe, _start_time = read_rigid_body_csv(path)
    marker_names = [get_rb_marker_name(index) for index in (5, 6, 4, 1)]
    positions = np.column_stack(
        [
            dataframe[[marker["x"] for marker in marker_names]].mean(axis=1),
            dataframe[[marker["y"] for marker in marker_names]].mean(axis=1),
            dataframe[[marker["z"] for marker in marker_names]].mean(axis=1),
        ]
    ).astype(np.float64)
    quaternions = dataframe[
        ["rb_ang_x", "rb_ang_y", "rb_ang_z", "rb_ang_w"]
    ].to_numpy(dtype=np.float64)
    times = dataframe["seconds"].to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(times)
        & np.isfinite(positions).all(axis=1)
        & np.isfinite(quaternions).all(axis=1)
        & (np.linalg.norm(quaternions, axis=1) > 1e-12)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"Not enough valid mocap samples in {path}")

    times = times[valid]
    times -= times[0]
    positions = positions[valid]
    rotations = Rotation.from_quat(
        quaternions[valid] / np.linalg.norm(quaternions[valid], axis=1)[:, None]
    )
    base_rotation = rotations[0].as_matrix()
    positions = (base_rotation.T @ (positions - positions[0]).T).T
    return MocapTrajectory(times, positions, rotations)


def inspect_reference(
    config: BenchmarkConfig,
) -> tuple[ReferenceInspection, Calibration, list[tuple[int, float]], MocapTrajectory]:
    required = (
        config.frames_path,
        config.timestamps_path,
        config.mocap_path,
        config.calibration_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing reference inputs: " + ", ".join(missing))

    frame_count = 0
    first_frame: np.ndarray | None = None
    for frame in iter_msgpack(config.frames_path):
        frame_count += 1
        if first_frame is None and isinstance(frame, np.ndarray):
            first_frame = frame
    if first_frame is None or first_frame.ndim != 2:
        raise ValueError("Reference stream does not contain monochrome ndarray frames")

    timestamps = read_timestamps(config.timestamps_path)
    if len(timestamps) != frame_count:
        raise ValueError(
            f"Frame/timestamp count mismatch: {frame_count} frames, "
            f"{len(timestamps)} timestamps"
        )
    active_count = sum(flag == 1 for flag, _timestamp in timestamps)
    if active_count == 0:
        raise ValueError("Reference timestamps contain no active sync interval")

    calibration = load_calibration(config.calibration_path)
    mocap = read_mocap(config.mocap_path)
    inspection = ReferenceInspection(
        frame_count=frame_count,
        timestamp_count=len(timestamps),
        active_frame_count=active_count,
        frame_shape=tuple(int(value) for value in first_frame.shape),
        frame_dtype=str(first_frame.dtype),
        mocap_sample_count=len(mocap.times_s),
        mocap_duration_s=float(mocap.times_s[-1] - mocap.times_s[0]),
        calibration_method=calibration.method,
        marker_length_m=calibration.marker_length,
    )
    return inspection, calibration, timestamps, mocap

