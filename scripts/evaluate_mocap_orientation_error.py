from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
import pandas as pd
from cv2 import aruco
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evaluate_rotation_smoothness import (
    baseline_pose,
    flatten_vector,
    make_undistorter,
    parse_timestamp,
    summarize_timing,
)
from pd_support import add_datetime_col, get_rb_marker_name, read_rigid_body_csv
from line_refine_single_frame import load_calibration, refine_tag_from_internal_lines


def iter_msgpack(path: Path):
    with path.open("rb") as file:
        yield from mp.Unpacker(file, object_hook=mpn.decode, raw=False)


def read_webcam_timestamps(path: Path) -> list[tuple[int, float]]:
    timestamps: list[tuple[int, float]] = []
    for item in iter_msgpack(path):
        if not isinstance(item, list | tuple) or len(item) < 2:
            continue
        timestamp = parse_timestamp(item[1])
        if timestamp is None:
            continue
        timestamps.append((int(item[0]), timestamp))
    return timestamps


def read_mocap_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    mocap_df, st_time = read_rigid_body_csv(path)
    mocap_df = add_datetime_col(mocap_df, st_time, "seconds")

    marker_names = [
        get_rb_marker_name(5),
        get_rb_marker_name(6),
        get_rb_marker_name(4),
        get_rb_marker_name(1),
    ]
    marker_positions = np.column_stack(
        [
            mocap_df[[marker_names[0]["x"], marker_names[1]["x"], marker_names[2]["x"], marker_names[3]["x"]]].to_numpy().mean(axis=1),
            mocap_df[[marker_names[0]["y"], marker_names[1]["y"], marker_names[2]["y"], marker_names[3]["y"]]].to_numpy().mean(axis=1),
            mocap_df[[marker_names[0]["z"], marker_names[1]["z"], marker_names[2]["z"], marker_names[3]["z"]]].to_numpy().mean(axis=1),
        ]
    )

    quaternions_xyzw = mocap_df[["rb_ang_x", "rb_ang_y", "rb_ang_z", "rb_ang_w"]].to_numpy(dtype=np.float64)
    quat_norms = np.linalg.norm(quaternions_xyzw, axis=1)
    valid = quat_norms > 1e-12
    if not np.any(valid):
        raise ValueError(f"Not enough valid mocap rotations in {path}")
    quaternions_xyzw = quaternions_xyzw[valid] / quat_norms[valid][:, None]
    positions_xyz = marker_positions[valid]
    times = pd.to_datetime(mocap_df.loc[valid, "time"]).astype("int64") / 1e9
    times = np.asarray(times, dtype=np.float64)
    times = times - times[0]

    rotation = Rotation.from_quat(quaternions_xyzw)
    base_rotation = rotation[0].as_matrix()
    base_position = positions_xyz[0]
    transformed_positions = (base_rotation.T @ (positions_xyz - base_position).T).T

    return times, transformed_positions, rotation


def rotation_from_rvec(rvec: np.ndarray | None) -> Rotation | None:
    if rvec is None:
        return None
    return Rotation.from_rotvec(rvec.reshape(3))


def euler_dict(prefix: str, values: np.ndarray | None) -> dict[str, float | None]:
    if values is None:
        return {f"{prefix}_euler_x_deg": None, f"{prefix}_euler_y_deg": None, f"{prefix}_euler_z_deg": None}
    return {
        f"{prefix}_euler_x_deg": float(values[0]),
        f"{prefix}_euler_y_deg": float(values[1]),
        f"{prefix}_euler_z_deg": float(values[2]),
    }


def relative_euler_series(rows: list[dict[str, Any]], prefix: str, euler_order: str) -> dict[int, np.ndarray]:
    indexed: list[tuple[int, Rotation]] = [
        (index, row[f"{prefix}_rotation"]) for index, row in enumerate(rows) if row.get(f"{prefix}_rotation") is not None
    ]
    if not indexed:
        return {}

    eulers_rad = np.asarray([rotation.as_euler(euler_order, degrees=False) for _, rotation in indexed])
    eulers_rad = np.unwrap(eulers_rad, axis=0)
    relative = eulers_rad - eulers_rad[0]
    relative_deg = np.degrees(relative)
    return {index: relative_deg[position] for position, (index, _rotation) in enumerate(indexed)}


def add_euler_trajectory_columns(rows: list[dict[str, Any]], euler_order: str) -> None:
    for prefix in ["mocap", "baseline", "refined"]:
        relative = relative_euler_series(rows, prefix, euler_order)
        for index, row in enumerate(rows):
            rotation = row.get(f"{prefix}_rotation")
            raw = rotation.as_euler(euler_order, degrees=True) if rotation is not None else None
            row.update(euler_dict(prefix, raw))
            row.update(euler_dict(f"{prefix}_rel", relative.get(index)))


def add_mocap_error_columns(rows: list[dict[str, Any]]) -> None:
    first_rotations: dict[str, tuple[Rotation, Rotation]] = {}
    previous: dict[str, tuple[Rotation, Rotation]] = {}

    for row in rows:
        mocap_rotation = row.get("mocap_rotation")
        for prefix in ["baseline", "refined"]:
            for axis in ["x", "y", "z"]:
                row[f"{prefix}_rel_euler_error_{axis}_deg"] = None
                row[f"{prefix}_delta_euler_error_{axis}_deg"] = None
            row[f"{prefix}_delta_angle_error_deg"] = None
            row[f"{prefix}_delta_angle_abs_error_deg"] = None
            row[f"{prefix}_calibrated_attitude_error_deg"] = None

            tag_rotation = row.get(f"{prefix}_rotation")
            if mocap_rotation is None or tag_rotation is None:
                previous.pop(prefix, None)
                continue

            if prefix not in first_rotations:
                first_rotations[prefix] = (mocap_rotation, tag_rotation)

            first_mocap, first_tag = first_rotations[prefix]
            left_alignment = first_tag * first_mocap.inv()
            expected_tag = left_alignment * mocap_rotation
            row[f"{prefix}_calibrated_attitude_error_deg"] = float((expected_tag.inv() * tag_rotation).magnitude() * 180.0 / np.pi)

            for axis in ["x", "y", "z"]:
                tag_value = row.get(f"{prefix}_rel_euler_{axis}_deg")
                mocap_value = row.get(f"mocap_rel_euler_{axis}_deg")
                if tag_value is not None and mocap_value is not None:
                    row[f"{prefix}_rel_euler_error_{axis}_deg"] = float(tag_value - mocap_value)

            if prefix in previous:
                prev_mocap, prev_tag = previous[prefix]
                mocap_delta = prev_mocap.inv() * mocap_rotation
                tag_delta = prev_tag.inv() * tag_rotation
                mocap_delta_angle = mocap_delta.magnitude() * 180.0 / np.pi
                tag_delta_angle = tag_delta.magnitude() * 180.0 / np.pi
                row[f"{prefix}_delta_angle_error_deg"] = float(tag_delta_angle - mocap_delta_angle)
                row[f"{prefix}_delta_angle_abs_error_deg"] = float(abs(tag_delta_angle - mocap_delta_angle))

                for axis in ["x", "y", "z"]:
                    tag_value = row.get(f"{prefix}_rel_euler_{axis}_deg")
                    mocap_value = row.get(f"mocap_rel_euler_{axis}_deg")
                    prev_tag_value = previous[prefix + f"_tag_rel_{axis}"]
                    prev_mocap_value = previous[prefix + f"_mocap_rel_{axis}"]
                    if tag_value is not None and mocap_value is not None:
                        row[f"{prefix}_delta_euler_error_{axis}_deg"] = float(
                            (tag_value - prev_tag_value) - (mocap_value - prev_mocap_value)
                        )

            previous[prefix] = (mocap_rotation, tag_rotation)
            for axis in ["x", "y", "z"]:
                previous[prefix + f"_tag_rel_{axis}"] = row.get(f"{prefix}_rel_euler_{axis}_deg")
                previous[prefix + f"_mocap_rel_{axis}"] = row.get(f"mocap_rel_euler_{axis}_deg")


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [row[key] for row in rows if row.get(key) is not None and np.isfinite(row[key])]
    return np.asarray(values, dtype=np.float64)


def rmse(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return float(np.sqrt(np.mean(values * values)))


def summarize_errors(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {
        "valid_frames": int(sum(row.get(f"{prefix}_ok", False) for row in rows)),
        "calibrated_attitude_error_rmse_deg": rmse(finite_values(rows, f"{prefix}_calibrated_attitude_error_deg")),
        "calibrated_attitude_error_median_deg": None,
        "delta_angle_abs_error_mean_deg": None,
        "delta_angle_abs_error_median_deg": None,
        "delta_angle_error_rmse_deg": rmse(finite_values(rows, f"{prefix}_delta_angle_error_deg")),
    }

    attitude = finite_values(rows, f"{prefix}_calibrated_attitude_error_deg")
    delta_abs = finite_values(rows, f"{prefix}_delta_angle_abs_error_deg")
    if len(attitude):
        summary["calibrated_attitude_error_median_deg"] = float(np.median(attitude))
    if len(delta_abs):
        summary["delta_angle_abs_error_mean_deg"] = float(np.mean(delta_abs))
        summary["delta_angle_abs_error_median_deg"] = float(np.median(delta_abs))

    for axis in ["x", "y", "z"]:
        summary[f"rel_euler_{axis}_rmse_deg"] = rmse(finite_values(rows, f"{prefix}_rel_euler_error_{axis}_deg"))
        summary[f"delta_euler_{axis}_rmse_deg"] = rmse(finite_values(rows, f"{prefix}_delta_euler_error_{axis}_deg"))

    return summary


def translation_rows(rows: list[dict[str, Any]], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    positions: list[list[float]] = []
    for row in rows:
        rotation = row.get(f"{prefix}_rotation")
        tvec_x = row.get(f"{prefix}_tvec_x")
        tvec_y = row.get(f"{prefix}_tvec_y")
        tvec_z = row.get(f"{prefix}_tvec_z")
        if rotation is None:
            continue
        if any(value is None or not np.isfinite(value) for value in [tvec_x, tvec_y, tvec_z]):
            continue
        times.append(float(row["active_time_s"]))
        positions.append([float(tvec_x), float(tvec_y), float(tvec_z)])

    if not times:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64).reshape(0, 3)
    return np.asarray(times, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def rebase_translation_to_first_frame(positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return positions
    return positions - positions[0]


def kabsch_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    source_centered = source - centroid_source
    target_centered = target - centroid_target
    h = source_centered.T @ target_centered
    u, _s, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[2, :] *= -1
        rotation = vt.T @ u.T
    translation = centroid_target - rotation @ centroid_source
    return rotation, translation


def rigid_align_points(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(source) != len(target) or len(source) == 0:
        raise ValueError("source and target must have matching non-empty lengths")

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    h = source_centered.T @ target_centered
    u, _s, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def align_translation_to_mocap(camera_positions: np.ndarray, mocap_positions: np.ndarray) -> np.ndarray:
    if len(camera_positions) == 0 or len(mocap_positions) == 0:
        return np.asarray([], dtype=np.float64).reshape(0, 3)
    n = min(len(camera_positions), len(mocap_positions))
    rotation, translation = rigid_align_points(camera_positions[:n], mocap_positions[:n])
    return (rotation @ camera_positions.T).T + translation


def plot_translation_comparison(
    rows: list[dict[str, Any]],
    plot_dir: Path,
    prefix: str,
    mocap_positions: np.ndarray,
) -> dict[str, np.ndarray]:
    times, positions = translation_rows(rows, prefix)
    if len(times) == 0 or len(mocap_positions) == 0:
        return {"times": times, "positions": positions, "aligned_mocap": np.asarray([], dtype=np.float64).reshape(0, 3)}

    positions = rebase_translation_to_first_frame(positions)
    clean_mask = np.isfinite(positions).all(axis=1) & np.isfinite(mocap_positions[: len(positions)]).all(axis=1)
    clean_tag = positions[clean_mask]
    clean_mocap = mocap_positions[: len(positions)][clean_mask]
    if len(clean_tag) == 0 or len(clean_mocap) == 0:
        return {"times": times, "positions": positions, "aligned_mocap": np.asarray([], dtype=np.float64).reshape(0, 3)}

    rotation, translation = kabsch_align(clean_mocap, clean_tag)
    aligned_mocap = (rotation @ mocap_positions[: len(positions)].T).T + translation

    n = min(len(times), len(aligned_mocap))
    times = times[:n]
    positions = positions[:n]
    aligned_mocap = aligned_mocap[:n]

    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axis_specs = [
        ("x", 0, "tab:blue", "tab:orange"),
        ("y", 1, "tab:green", "tab:red"),
        ("z", 2, "tab:purple", "tab:brown"),
    ]
    for axis_label, axis_index, cam_color, mocap_color in axis_specs:
        axes[axis_index].plot(times, positions[:, axis_index], label=f"{prefix} tvec {axis_label}", color=cam_color)
        axes[axis_index].plot(
            times,
            aligned_mocap[:, axis_index],
            label=f"mocap {axis_label}",
            color=mocap_color,
            linestyle="--",
        )
        axes[axis_index].set_ylabel(f"{axis_label} (m)")
        axes[axis_index].grid(True, alpha=0.3)
        axes[axis_index].legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(f"{prefix.capitalize()} translation vs mocap")
    figure.tight_layout()
    figure.savefig(plot_dir / f"{prefix}_translation_vs_mocap.png", dpi=160)
    plt.close(figure)

    return {"times": times, "positions": positions, "aligned_mocap": aligned_mocap}


def plot_xy_xz_trajectories(
    plot_dir: Path,
    baseline_positions: np.ndarray,
    refined_positions: np.ndarray,
    baseline_mocap: np.ndarray,
    refined_mocap: np.ndarray,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].plot(baseline_positions[:, 0], baseline_positions[:, 1], label="baseline cam", color="tab:blue")
    axes[0].plot(baseline_mocap[:, 0], baseline_mocap[:, 1], label="baseline mocap", color="tab:orange", linestyle="--")
    axes[0].plot(refined_positions[:, 0], refined_positions[:, 1], label="refined cam", color="tab:green")
    axes[0].plot(refined_mocap[:, 0], refined_mocap[:, 1], label="refined mocap", color="tab:red", linestyle="--")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("XY trajectory")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(baseline_positions[:, 0], baseline_positions[:, 2], label="baseline cam", color="tab:blue")
    axes[1].plot(baseline_mocap[:, 0], baseline_mocap[:, 2], label="baseline mocap", color="tab:orange", linestyle="--")
    axes[1].plot(refined_positions[:, 0], refined_positions[:, 2], label="refined cam", color="tab:green")
    axes[1].plot(refined_mocap[:, 0], refined_mocap[:, 2], label="refined mocap", color="tab:red", linestyle="--")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    axes[1].set_title("XZ trajectory")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    figure.tight_layout()
    figure.savefig(plot_dir / "trajectory_xy_xz_comparison.png", dpi=160)
    plt.close(figure)


def summarize_translation_error(camera_positions: np.ndarray, mocap_positions: np.ndarray) -> dict[str, float | None]:
    if len(camera_positions) == 0 or len(mocap_positions) == 0:
        return {
            "mean_abs_error_m": None,
            "rmse_m": None,
            "median_abs_error_m": None,
        }
    n = min(len(camera_positions), len(mocap_positions))
    error = camera_positions[:n] - mocap_positions[:n]
    distances = np.linalg.norm(error, axis=1)
    return {
        "mean_abs_error_m": float(np.mean(distances)),
        "rmse_m": float(np.sqrt(np.mean(distances * distances))),
        "median_abs_error_m": float(np.median(distances)),
    }


def serializable_row(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in fieldnames}


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "active_time_s",
        "webcam_timestamp_s",
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
        "refined_ok",
        "line_count",
        "intersection_count",
        "baseline_reprojection_rms_px",
        "refined_reprojection_rms_px",
    ]

    for prefix in ["mocap", "baseline", "baseline_rel", "refined", "refined_rel", "mocap_rel"]:
        fieldnames.extend([f"{prefix}_euler_x_deg", f"{prefix}_euler_y_deg", f"{prefix}_euler_z_deg"])

    for prefix in ["baseline", "refined"]:
        fieldnames.extend(
            [
                f"{prefix}_rel_euler_error_x_deg",
                f"{prefix}_rel_euler_error_y_deg",
                f"{prefix}_rel_euler_error_z_deg",
                f"{prefix}_delta_euler_error_x_deg",
                f"{prefix}_delta_euler_error_y_deg",
                f"{prefix}_delta_euler_error_z_deg",
                f"{prefix}_delta_angle_error_deg",
                f"{prefix}_delta_angle_abs_error_deg",
                f"{prefix}_calibrated_attitude_error_deg",
                f"{prefix}_rvec_x",
                f"{prefix}_rvec_y",
                f"{prefix}_rvec_z",
                f"{prefix}_tvec_x",
                f"{prefix}_tvec_y",
                f"{prefix}_tvec_z",
            ]
        )

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(serializable_row(row, fieldnames))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AprilTag orientation trajectory against OptiTrack mocap.")
    default_recording = Path("data/ref_recording_single_april_diwakar")
    parser.add_argument("--data", type=Path, default=default_recording / "webcam_color.msgpack")
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=default_recording / "webcam_timestamp.msgpack",
    )
    parser.add_argument(
        "--mocap-csv",
        type=Path,
        default=default_recording / "ref_recording_single_april_diwakar.csv",
    )
    parser.add_argument("--calibration", type=Path, default=Path("calibration/diwakar_calibration.toml"))
    parser.add_argument("--tag-id", type=int, default=12)
    parser.add_argument("--max-active-frames", type=int, default=300, help="Use 0 for all active mocap frames.")
    parser.add_argument("--euler-order", default="xyz")
    parser.add_argument("--samples-per-segment", type=int, default=5)
    parser.add_argument("--profile-radius", type=int, default=5)
    parser.add_argument("--min-gradient", type=float, default=8.0)
    parser.add_argument("--min-line-points", type=int, default=5)
    parser.add_argument("--max-line-rms", type=float, default=2.5)
    parser.add_argument("--output", type=Path, default=default_recording / "mocap_orientation_error.csv")
    parser.add_argument("--plot-dir", type=Path, default=default_recording, help="Directory for generated PNG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    webcam_timestamps = read_webcam_timestamps(args.timestamps)
    active_start = next((timestamp for flag, timestamp in webcam_timestamps if flag == 1), None)
    if active_start is None:
        raise RuntimeError("No active mocap segment found in webcam timestamps")

    mocap_times, mocap_positions, mocap_rotations = read_mocap_trajectory(args.mocap_csv)
    mocap_slerp = Slerp(mocap_times, mocap_rotations)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    undistort = None
    camera_matrix = None
    rows: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(iter_msgpack(args.data)):
        if frame_index >= len(webcam_timestamps):
            break
        flag, webcam_timestamp = webcam_timestamps[frame_index]
        if flag != 1:
            continue
        if args.max_active_frames > 0 and len(rows) >= args.max_active_frames:
            break
        if not isinstance(frame, np.ndarray) or frame.ndim != 2:
            continue

        active_time = webcam_timestamp - active_start
        if active_time < mocap_times[0] or active_time > mocap_times[-1]:
            continue

        frame_started_at = time.perf_counter()
        if undistort is None or camera_matrix is None:
            undistort, camera_matrix = make_undistorter(frame.shape, calibration)

        undistort_started_at = time.perf_counter()
        gray = undistort(frame)
        undistort_ms = (time.perf_counter() - undistort_started_at) * 1000.0

        detection_started_at = time.perf_counter()
        corners_list, ids, _rejected = detector.detectMarkers(gray)
        detection_ms = (time.perf_counter() - detection_started_at) * 1000.0
        detected_ids = [] if ids is None else ids.ravel().astype(int).tolist()

        mocap_rotation = mocap_slerp([active_time])[0]
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "active_time_s": active_time,
            "webcam_timestamp_s": webcam_timestamp,
            "detected_ids": " ".join(str(marker_id) for marker_id in detected_ids),
            "selected_id": args.tag_id,
            "mocap_rotation": mocap_rotation,
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
            row["baseline_rotation"] = rotation_from_rvec(baseline_rvec) if baseline_ok else None
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
            row["refined_rotation"] = rotation_from_rvec(refined_rvec) if result["pose_ok"] else None
            row["refined_reprojection_rms_px"] = result["reprojection_rms"]
            row["line_count"] = len(lines) if isinstance(lines, dict) else 0
            row["intersection_count"] = len(image_points) if isinstance(image_points, np.ndarray) else 0
            row.update(flatten_vector("refined_rvec", refined_rvec))
            row.update(flatten_vector("refined_tvec", refined_tvec))
        else:
            row["baseline_rotation"] = None
            row["refined_rotation"] = None
            row.update(flatten_vector("baseline_rvec", None))
            row.update(flatten_vector("baseline_tvec", None))
            row.update(flatten_vector("refined_rvec", None))
            row.update(flatten_vector("refined_tvec", None))

        row["baseline_pipeline_ms"] = row["undistort_ms"] + row["detection_ms"] + row["baseline_pose_ms"]
        row["refined_pipeline_ms"] = row["baseline_pipeline_ms"] + row["refinement_ms"]
        row["total_frame_ms"] = (time.perf_counter() - frame_started_at) * 1000.0
        rows.append(row)

    add_euler_trajectory_columns(rows, args.euler_order)
    add_mocap_error_columns(rows)
    write_csv(rows, args.output)

    plot_dir = args.plot_dir if args.plot_dir is not None else args.output.parent
    baseline_plot = plot_translation_comparison(rows, plot_dir, "baseline", mocap_positions)
    refined_plot = plot_translation_comparison(rows, plot_dir, "refined", mocap_positions)
    if len(baseline_plot["positions"]) and len(refined_plot["positions"]):
        plot_xy_xz_trajectories(
            plot_dir,
            baseline_plot["positions"],
            refined_plot["positions"],
            baseline_plot["aligned_mocap"],
            refined_plot["aligned_mocap"],
        )

    print(f"plot_dir={plot_dir}")
    print(f"baseline_translation_error={summarize_translation_error(baseline_plot['positions'], baseline_plot['aligned_mocap'])}")
    print(f"refined_translation_error={summarize_translation_error(refined_plot['positions'], refined_plot['aligned_mocap'])}")

    print(f"active_frames_processed={len(rows)}")
    print(f"tag_id={args.tag_id}")
    print(f"active_start_webcam_frame={next(i for i, (flag, _timestamp) in enumerate(webcam_timestamps) if flag == 1)}")
    print(f"csv={args.output}")
    print(f"baseline_mocap_error={summarize_errors(rows, 'baseline')}")
    print(f"refined_mocap_error={summarize_errors(rows, 'refined')}")
    print(f"timing_ms={summarize_timing(rows)}")


if __name__ == "__main__":
    main()
