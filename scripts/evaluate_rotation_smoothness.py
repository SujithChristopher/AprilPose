from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
from cv2 import aruco

from line_refine_single_frame import (
    Calibration,
    grid_to_object_points,
    load_calibration,
    refine_tag_from_internal_lines,
    solve_pose,
)


def iter_msgpack(path: Path):
    with path.open("rb") as file:
        yield from mp.Unpacker(file, object_hook=mpn.decode, raw=False)


def parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return datetime.fromisoformat(value).timestamp()
    if isinstance(value, list | tuple) and value:
        return parse_timestamp(value[-1])
    return None


def make_undistorter(frame_shape: tuple[int, int], calibration: Calibration):
    if calibration.method != "fisheye":
        return lambda frame: frame, calibration.camera_matrix

    height, width = frame_shape
    image_size = (width, height)
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
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
        new_camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )

    def undistort(frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    return undistort, new_camera_matrix


def baseline_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    marker_length: float,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None]:
    object_grid = np.asarray([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]], dtype=np.float64)
    object_points = grid_to_object_points(object_grid, grid_size=8, marker_length=marker_length)
    return solve_pose(object_points, corners.astype(np.float64), camera_matrix)


def rotation_matrix(rvec: np.ndarray | None) -> np.ndarray | None:
    if rvec is None:
        return None
    matrix, _ = cv2.Rodrigues(rvec)
    return matrix


def relative_angle_deg(prev_rvec: np.ndarray | None, curr_rvec: np.ndarray | None) -> float | None:
    prev_matrix = rotation_matrix(prev_rvec)
    curr_matrix = rotation_matrix(curr_rvec)
    if prev_matrix is None or curr_matrix is None:
        return None
    relative = curr_matrix @ prev_matrix.T
    cos_theta = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def flatten_vector(prefix: str, vector: np.ndarray | None) -> dict[str, float | None]:
    if vector is None:
        return {f"{prefix}_x": None, f"{prefix}_y": None, f"{prefix}_z": None}
    values = vector.reshape(-1)
    return {
        f"{prefix}_x": float(values[0]),
        f"{prefix}_y": float(values[1]),
        f"{prefix}_z": float(values[2]),
    }


def add_rotation_metrics(rows: list[dict[str, Any]], prefix: str) -> None:
    prev_valid_row: dict[str, Any] | None = None
    prev_velocity: float | None = None

    for row in rows:
        row[f"{prefix}_delta_angle_deg"] = None
        row[f"{prefix}_angular_velocity_deg_s"] = None
        row[f"{prefix}_angular_accel_deg_s2"] = None

        rvec = row.get(f"{prefix}_rvec")
        if rvec is None:
            prev_valid_row = None
            prev_velocity = None
            continue

        if prev_valid_row is None:
            prev_valid_row = row
            continue

        prev_rvec = prev_valid_row.get(f"{prefix}_rvec")
        delta_angle = relative_angle_deg(prev_rvec, rvec)
        dt = row["time_s"] - prev_valid_row["time_s"]
        if delta_angle is None or dt <= 0:
            prev_valid_row = row
            prev_velocity = None
            continue

        velocity = delta_angle / dt
        row[f"{prefix}_delta_angle_deg"] = delta_angle
        row[f"{prefix}_angular_velocity_deg_s"] = velocity

        if prev_velocity is not None:
            row[f"{prefix}_angular_accel_deg_s2"] = (velocity - prev_velocity) / dt

        prev_valid_row = row
        prev_velocity = velocity


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [row[key] for row in rows if row.get(key) is not None and np.isfinite(row[key])]
    return np.asarray(values, dtype=np.float64)


def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | int | None]:
    delta_angle = finite_values(rows, f"{prefix}_delta_angle_deg")
    velocity = finite_values(rows, f"{prefix}_angular_velocity_deg_s")
    accel = finite_values(rows, f"{prefix}_angular_accel_deg_s2")
    reproj = finite_values(rows, f"{prefix}_reprojection_rms_px")

    def rms(values: np.ndarray) -> float | None:
        if len(values) == 0:
            return None
        return float(np.sqrt(np.mean(values * values)))

    def std(values: np.ndarray) -> float | None:
        if len(values) == 0:
            return None
        return float(np.std(values))

    def median_abs(values: np.ndarray) -> float | None:
        if len(values) == 0:
            return None
        return float(np.median(np.abs(values)))

    return {
        "valid_frames": int(sum(row.get(f"{prefix}_ok", False) for row in rows)),
        "delta_angle_std_deg": std(delta_angle),
        "delta_angle_median_abs_deg": median_abs(delta_angle),
        "velocity_std_deg_s": std(velocity),
        "velocity_rms_deg_s": rms(velocity),
        "accel_std_deg_s2": std(accel),
        "accel_rms_deg_s2": rms(accel),
        "accel_median_abs_deg_s2": median_abs(accel),
        "reprojection_rms_px_mean": float(np.mean(reproj)) if len(reproj) else None,
        "reprojection_rms_px_median": float(np.median(reproj)) if len(reproj) else None,
    }


def summarize_timing(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    timing_keys = [
        "undistort_ms",
        "detection_ms",
        "baseline_pose_ms",
        "refinement_ms",
        "baseline_pipeline_ms",
        "refined_pipeline_ms",
        "total_frame_ms",
    ]

    summary: dict[str, dict[str, float | None]] = {}
    for key in timing_keys:
        values = finite_values(rows, key)
        if len(values) == 0:
            summary[key] = {"mean": None, "median": None, "p95": None}
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    return summary


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "time_s",
        "detected_ids",
        "selected_id",
        "undistort_ms",
        "detection_ms",
        "baseline_pose_ms",
        "refinement_ms",
        "baseline_pipeline_ms",
        "refined_pipeline_ms",
        "total_frame_ms",
        "baseline_ok",
        "baseline_reprojection_rms_px",
        "baseline_delta_angle_deg",
        "baseline_angular_velocity_deg_s",
        "baseline_angular_accel_deg_s2",
        "baseline_rvec_x",
        "baseline_rvec_y",
        "baseline_rvec_z",
        "baseline_tvec_x",
        "baseline_tvec_y",
        "baseline_tvec_z",
        "refined_ok",
        "refined_reprojection_rms_px",
        "refined_delta_angle_deg",
        "refined_angular_velocity_deg_s",
        "refined_angular_accel_deg_s2",
        "refined_rvec_x",
        "refined_rvec_y",
        "refined_rvec_z",
        "refined_tvec_x",
        "refined_tvec_y",
        "refined_tvec_z",
        "line_count",
        "intersection_count",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {key: row.get(key) for key in fieldnames}
            writer.writerow(serializable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs internal-line AprilTag rotation smoothness.")
    parser.add_argument("--data", type=Path, default=Path("data/3marker_april_mono_160fov_3/webcam_color.msgpack"))
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=Path("data/3marker_april_mono_160fov_3/webcam_timestamp.msgpack"),
    )
    parser.add_argument("--calibration", type=Path, default=Path("calibration/good.toml"))
    parser.add_argument("--tag-id", type=int, default=14)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=300, help="Number of frames to process; use 0 for all frames.")
    parser.add_argument("--samples-per-segment", type=int, default=5)
    parser.add_argument("--profile-radius", type=int, default=5)
    parser.add_argument("--min-gradient", type=float, default=8.0)
    parser.add_argument("--min-line-points", type=int, default=5)
    parser.add_argument("--max-line-rms", type=float, default=2.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/rotation_smoothness.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    timestamp_iter = iter(iter_msgpack(args.timestamps)) if args.timestamps.exists() else None
    undistort = None
    camera_matrix = None
    rows: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(iter_msgpack(args.data)):
        frame_started_at = time.perf_counter()
        timestamp_value = next(timestamp_iter, None) if timestamp_iter is not None else None
        if frame_index < args.start_frame:
            continue
        if args.max_frames > 0 and len(rows) >= args.max_frames:
            break
        if not isinstance(frame, np.ndarray) or frame.ndim != 2:
            continue

        if undistort is None or camera_matrix is None:
            undistort, camera_matrix = make_undistorter(frame.shape, calibration)

        undistort_started_at = time.perf_counter()
        gray = undistort(frame)
        undistort_ms = (time.perf_counter() - undistort_started_at) * 1000.0
        timestamp_s = parse_timestamp(timestamp_value)
        if timestamp_s is None:
            timestamp_s = float(frame_index)

        detection_started_at = time.perf_counter()
        corners_list, ids, _rejected = detector.detectMarkers(gray)
        detection_ms = (time.perf_counter() - detection_started_at) * 1000.0
        detected_ids = [] if ids is None else ids.ravel().astype(int).tolist()
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "time_s": timestamp_s,
            "detected_ids": " ".join(str(marker_id) for marker_id in detected_ids),
            "selected_id": args.tag_id,
            "undistort_ms": undistort_ms,
            "detection_ms": detection_ms,
            "baseline_pose_ms": 0.0,
            "refinement_ms": 0.0,
            "baseline_ok": False,
            "refined_ok": False,
            "line_count": 0,
            "intersection_count": 0,
        }

        if ids is not None and args.tag_id in detected_ids:
            tag_index = detected_ids.index(args.tag_id)
            corners = corners_list[tag_index].reshape(4, 2)

            baseline_started_at = time.perf_counter()
            baseline_ok, baseline_rvec, baseline_tvec, baseline_rms = baseline_pose(
                corners,
                camera_matrix,
                calibration.marker_length,
            )
            row["baseline_pose_ms"] = (time.perf_counter() - baseline_started_at) * 1000.0
            row["baseline_ok"] = baseline_ok
            row["baseline_rvec"] = baseline_rvec
            row["baseline_reprojection_rms_px"] = baseline_rms
            row.update(flatten_vector("baseline_rvec", baseline_rvec))
            row.update(flatten_vector("baseline_tvec", baseline_tvec))

            refinement_started_at = time.perf_counter()
            result = refine_tag_from_internal_lines(
                gray,
                corners,
                args.tag_id,
                dictionary,
                camera_matrix,
                calibration.marker_length,
                samples_per_segment=args.samples_per_segment,
                profile_radius=args.profile_radius,
                min_gradient=args.min_gradient,
                min_line_points=args.min_line_points,
                max_line_rms=args.max_line_rms,
            )
            row["refinement_ms"] = (time.perf_counter() - refinement_started_at) * 1000.0
            refined_rvec = result["rvec"] if result["pose_ok"] else None
            refined_tvec = result["tvec"] if result["pose_ok"] else None
            image_points = result["image_points"]
            lines = result["lines"]

            row["refined_ok"] = bool(result["pose_ok"])
            row["refined_rvec"] = refined_rvec
            row["refined_reprojection_rms_px"] = result["reprojection_rms"]
            row["line_count"] = len(lines) if isinstance(lines, dict) else 0
            row["intersection_count"] = len(image_points) if isinstance(image_points, np.ndarray) else 0
            row.update(flatten_vector("refined_rvec", refined_rvec))
            row.update(flatten_vector("refined_tvec", refined_tvec))
        else:
            row.update(flatten_vector("baseline_rvec", None))
            row.update(flatten_vector("baseline_tvec", None))
            row.update(flatten_vector("refined_rvec", None))
            row.update(flatten_vector("refined_tvec", None))

        row["baseline_pipeline_ms"] = row["undistort_ms"] + row["detection_ms"] + row["baseline_pose_ms"]
        row["refined_pipeline_ms"] = row["baseline_pipeline_ms"] + row["refinement_ms"]
        row["total_frame_ms"] = (time.perf_counter() - frame_started_at) * 1000.0
        rows.append(row)

    add_rotation_metrics(rows, "baseline")
    add_rotation_metrics(rows, "refined")
    write_csv(rows, args.output)

    baseline_summary = summarize(rows, "baseline")
    refined_summary = summarize(rows, "refined")
    print(f"frames_processed={len(rows)}")
    print(f"tag_id={args.tag_id}")
    print(f"csv={args.output}")
    print(f"baseline={baseline_summary}")
    print(f"refined={refined_summary}")
    print(f"timing_ms={summarize_timing(rows)}")


if __name__ == "__main__":
    main()
