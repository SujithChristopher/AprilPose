from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2 import aruco
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp

from .config import BenchmarkConfig
from .metrics import (
    orientation_change_summary,
    timing_summary,
    translation_summary,
)
from .models import PoseModel, PoseResult, build_models
from .reference import inspect_reference, iter_msgpack


def _make_undistorter(frame_shape, calibration):
    if calibration.method != "fisheye":
        return lambda frame: frame, calibration.camera_matrix

    height, width = frame_shape
    image_size = (width, height)
    camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        calibration.camera_matrix,
        calibration.dist_coeffs,
        image_size,
        np.eye(3),
        balance=1.0,
        new_size=image_size,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        calibration.camera_matrix,
        calibration.dist_coeffs,
        np.eye(3),
        camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )

    def undistort(frame):
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    return undistort, camera_matrix


def _detector(marker_length: float):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    board = aruco.GridBoard(
        size=[1, 1],
        markerLength=marker_length,
        markerSeparation=0.01,
        dictionary=dictionary,
    )
    return dictionary, detector, board


def _detect(
    gray: np.ndarray,
    detector: aruco.ArucoDetector,
    board: aruco.GridBoard,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    corners, ids, rejected = detector.detectMarkers(gray)
    corners, ids, _rejected, _recovered = detector.refineDetectedMarkers(
        gray,
        board,
        corners,
        ids,
        rejected,
    )
    return corners, ids


def _validate_result(model: PoseModel, result: PoseResult) -> dict[str, Any]:
    shape_ok = (
        result.rvec is not None
        and result.tvec is not None
        and np.asarray(result.rvec).size == 3
        and np.asarray(result.tvec).size == 3
    )
    finite = (
        shape_ok
        and np.isfinite(result.rvec).all()
        and np.isfinite(result.tvec).all()
    )
    positive_depth = finite and float(np.asarray(result.tvec).reshape(3)[2]) > 0.0
    return {
        "model": model.name,
        "ok": bool(result.ok and shape_ok and finite and positive_depth),
        "pose_ok": result.ok,
        "shape_ok": bool(shape_ok),
        "finite": bool(finite),
        "positive_depth": bool(positive_depth),
        "metadata": result.metadata,
    }


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _check_models(
    config: BenchmarkConfig,
    calibration,
    models: list[PoseModel],
    dictionary,
    detector,
    board,
) -> list[dict[str, Any]]:
    undistort = None
    camera_matrix = None
    for frame_index, frame in enumerate(iter_msgpack(config.frames_path)):
        if not isinstance(frame, np.ndarray) or frame.ndim != 2:
            continue
        if undistort is None or camera_matrix is None:
            undistort, camera_matrix = _make_undistorter(
                frame.shape,
                calibration,
            )
        gray = undistort(frame)
        corners_list, ids = _detect(gray, detector, board)
        detected_ids = [] if ids is None else ids.ravel().astype(int).tolist()
        if config.tag_id not in detected_ids:
            continue

        corners = corners_list[detected_ids.index(config.tag_id)].reshape(4, 2)
        checks: list[dict[str, Any]] = []
        for model in models:
            model.reset()
            result = model.predict(
                gray,
                corners,
                config.tag_id,
                dictionary,
                camera_matrix,
                calibration.marker_length,
            )
            check = _validate_result(model, result)
            check["frame_index"] = frame_index
            checks.append(check)
            model.reset()
        return checks

    raise RuntimeError(
        f"Could not check models because tag {config.tag_id} was not detected"
    )


def run_pipeline(config: BenchmarkConfig) -> dict[str, Any]:
    inspection, calibration, timestamps, mocap = inspect_reference(config)
    models = build_models(config)
    dictionary, detector, board = _detector(calibration.marker_length)
    model_checks = _check_models(
        config,
        calibration,
        models,
        dictionary,
        detector,
        board,
    )
    failed_checks = [check["model"] for check in model_checks if not check["ok"]]
    if failed_checks:
        raise RuntimeError("Model checks failed: " + ", ".join(failed_checks))

    undistort = None
    camera_matrix = None
    active_start = next(timestamp for flag, timestamp in timestamps if flag == 1)
    rows: list[dict[str, Any]] = []
    model_rvecs: dict[str, list[np.ndarray | None]] = {
        model.name: [] for model in models
    }
    model_tvecs: dict[str, list[np.ndarray | None]] = {
        model.name: [] for model in models
    }
    model_timings: dict[str, list[float]] = {model.name: [] for model in models}
    active_times: list[float] = []

    for model in models:
        model.reset()

    for frame_index, frame in enumerate(iter_msgpack(config.frames_path)):
        if not isinstance(frame, np.ndarray) or frame.ndim != 2:
            continue
        if undistort is None or camera_matrix is None:
            undistort, camera_matrix = _make_undistorter(
                frame.shape,
                calibration,
            )

        flag, timestamp = timestamps[frame_index]
        if flag != 1:
            continue
        active_time = timestamp - active_start
        if active_time < mocap.times_s[0] or active_time > mocap.times_s[-1]:
            continue
        if config.max_active_frames > 0 and len(rows) >= config.max_active_frames:
            break

        frame_started = time.perf_counter()
        gray = undistort(frame)
        detection_started = time.perf_counter()
        corners_list, ids = _detect(gray, detector, board)
        detection_ms = (time.perf_counter() - detection_started) * 1000.0
        detected_ids = [] if ids is None else ids.ravel().astype(int).tolist()
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "active_time_s": active_time,
            "detection_ms": detection_ms,
            "detected_ids": " ".join(map(str, detected_ids)),
        }
        target_corners = None
        if config.tag_id in detected_ids:
            target_corners = corners_list[detected_ids.index(config.tag_id)].reshape(4, 2)

        for model in models:
            result = PoseResult(False, None, None, None, {"reason": "tag_not_detected"})
            elapsed_ms = 0.0
            if target_corners is not None:
                started = time.perf_counter()
                result = model.predict(
                    gray,
                    target_corners,
                    config.tag_id,
                    dictionary,
                    camera_matrix,
                    calibration.marker_length,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            elif hasattr(model, "reset"):
                model.reset()

            rvec = result.rvec if result.ok else None
            tvec = result.tvec if result.ok else None
            model_rvecs[model.name].append(rvec)
            model_tvecs[model.name].append(tvec)
            if target_corners is not None:
                model_timings[model.name].append(elapsed_ms)
            row[f"{model.name}_ok"] = result.ok
            row[f"{model.name}_runtime_ms"] = elapsed_ms
            row[f"{model.name}_reprojection_rms_px"] = result.reprojection_rms_px
            for axis, value in zip(
                "xyz",
                np.asarray(rvec).reshape(3) if rvec is not None else [None] * 3,
                strict=True,
            ):
                row[f"{model.name}_rvec_{axis}"] = value
            for axis, value in zip(
                "xyz",
                np.asarray(tvec).reshape(3) if tvec is not None else [None] * 3,
                strict=True,
            ):
                row[f"{model.name}_tvec_{axis}"] = value
            for key, value in result.metadata.items():
                row[f"{model.name}_{key}"] = value

        row["total_frame_ms"] = (time.perf_counter() - frame_started) * 1000.0
        rows.append(row)
        active_times.append(active_time)

    if not rows:
        raise RuntimeError("No active reference frames were benchmarked")

    active_times_array = np.asarray(active_times, dtype=np.float64)
    mocap_positions = np.column_stack(
        [
            interp1d(mocap.times_s, mocap.positions_m[:, axis])(
                active_times_array
            )
            for axis in range(3)
        ]
    )
    mocap_rotations = Slerp(mocap.times_s, mocap.rotations)(active_times_array)

    models_summary: dict[str, Any] = {}
    for model in models:
        tvecs = np.full((len(rows), 3), np.nan, dtype=np.float64)
        for index, tvec in enumerate(model_tvecs[model.name]):
            if tvec is not None:
                tvecs[index] = np.asarray(tvec).reshape(3)
        models_summary[model.name] = {
            "valid_frames": int(np.isfinite(tvecs).all(axis=1).sum()),
            "translation": translation_summary(tvecs, mocap_positions),
            "orientation": orientation_change_summary(
                model_rvecs[model.name],
                mocap_rotations,
            ),
            "timing": timing_summary(model_timings[model.name]),
        }

    summary = {
        "config": {
            **asdict(config),
            "recording_dir": str(config.recording_dir),
            "calibration_path": str(config.calibration_path),
            "output_dir": str(config.output_dir),
            "model_names": list(config.model_names),
        },
        "inspection": inspection.to_dict(),
        "model_checks": model_checks,
        "frames_benchmarked": len(rows),
        "models": models_summary,
    }
    _write_rows(rows, config.output_dir / "frames.csv")
    _write_summary(summary, config.output_dir / "summary.json")
    return summary
