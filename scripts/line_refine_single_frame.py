from __future__ import annotations

import argparse
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
from cv2 import aruco

try:
    from aprilpose_rust import refine_internal_lines_rust
except ImportError:
    refine_internal_lines_rust = None

if os.environ.get("APRILPOSE_DISABLE_RUST") == "1":
    refine_internal_lines_rust = None


@dataclass(frozen=True)
class Calibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    method: str
    marker_length: float
    resolution: tuple[int, int]


@dataclass(frozen=True)
class EdgeSegment:
    orientation: Literal["vertical", "horizontal"]
    line_index: int
    first_color: int
    second_color: int
    p0_tag: np.ndarray
    p1_tag: np.ndarray


@dataclass(frozen=True)
class EdgePoint:
    point: np.ndarray
    weight: float


def unpack_frame(data_path: Path, frame_index: int) -> np.ndarray:
    with data_path.open("rb") as file:
        unpacker = mp.Unpacker(file, object_hook=mpn.decode, raw=False)
        for index, frame in enumerate(unpacker):
            if index == frame_index:
                if not isinstance(frame, np.ndarray):
                    raise TypeError(f"Expected ndarray frame, got {type(frame)!r}")
                if frame.ndim != 2:
                    raise ValueError(f"Expected monochrome frame, got shape {frame.shape}")
                return frame
    raise IndexError(f"Frame index {frame_index} is outside the msgpack stream")


def load_calibration(path: Path) -> Calibration:
    with path.open("rb") as file:
        data = tomllib.load(file)

    calibration = data["calibration"]
    aruco_data = data["aruco"]
    camera_data = data["camera"]

    return Calibration(
        camera_matrix=np.asarray(calibration["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.asarray(calibration["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
        method=str(calibration["method"]),
        marker_length=float(aruco_data["marker_length"]),
        resolution=tuple(camera_data["resolution"]),
    )


def undistort_frame(frame: np.ndarray, calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    if calibration.method != "fisheye":
        return frame, calibration.camera_matrix

    height, width = frame.shape[:2]
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
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted, new_camera_matrix


def build_marker_cell_grid(dictionary: cv2.aruco.Dictionary, marker_id: int, border_bits: int = 1) -> np.ndarray:
    marker_size = int(dictionary.markerSize)
    grid_size = marker_size + 2 * border_bits
    cell_px = 20
    image = aruco.generateImageMarker(dictionary, marker_id, grid_size * cell_px, borderBits=border_bits)
    cells = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * cell_px
            x0 = col * cell_px
            cell = image[y0 : y0 + cell_px, x0 : x0 + cell_px]
            cells[row, col] = 1 if float(cell.mean()) > 127.0 else 0
    return cells


def visible_edge_segments(cells: np.ndarray) -> list[EdgeSegment]:
    segments: list[EdgeSegment] = []
    rows, cols = cells.shape

    for col in range(1, cols):
        for row in range(rows):
            left = int(cells[row, col - 1])
            right = int(cells[row, col])
            if left == right:
                continue
            segments.append(
                EdgeSegment(
                    orientation="vertical",
                    line_index=col,
                    first_color=left,
                    second_color=right,
                    p0_tag=np.array([col, row], dtype=np.float64),
                    p1_tag=np.array([col, row + 1], dtype=np.float64),
                )
            )

    for row in range(1, rows):
        for col in range(cols):
            top = int(cells[row - 1, col])
            bottom = int(cells[row, col])
            if top == bottom:
                continue
            segments.append(
                EdgeSegment(
                    orientation="horizontal",
                    line_index=row,
                    first_color=top,
                    second_color=bottom,
                    p0_tag=np.array([col, row], dtype=np.float64),
                    p1_tag=np.array([col + 1, row], dtype=np.float64),
                )
            )

    return segments


def project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = points_h @ homography.T
    return projected[:, :2] / projected[:, 2:3]


def bilinear_sample(gray: np.ndarray, point: np.ndarray) -> float | None:
    x, y = float(point[0]), float(point[1])
    height, width = gray.shape[:2]
    if x < 0 or y < 0 or x >= width - 1 or y >= height - 1:
        return None

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    dx = x - x0
    dy = y - y0

    top = (1.0 - dx) * float(gray[y0, x0]) + dx * float(gray[y0, x0 + 1])
    bottom = (1.0 - dx) * float(gray[y0 + 1, x0]) + dx * float(gray[y0 + 1, x0 + 1])
    return (1.0 - dy) * top + dy * bottom


def subpixel_peak_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def detect_edge_points_for_segment(
    gray: np.ndarray,
    homography: np.ndarray,
    segment: EdgeSegment,
    samples_per_segment: int,
    profile_radius: int,
    min_gradient: float,
) -> list[EdgePoint]:
    projected = project_points(homography, np.vstack([segment.p0_tag, segment.p1_tag]))
    p0 = projected[0]
    p1 = projected[1]
    tangent = p1 - p0
    length = float(np.linalg.norm(tangent))
    if length < 2.0:
        return []

    tangent /= length
    segment_center_tag = 0.5 * (segment.p0_tag + segment.p1_tag)
    if segment.orientation == "vertical":
        normal_target_tag = segment_center_tag + np.array([0.1, 0.0], dtype=np.float64)
    else:
        normal_target_tag = segment_center_tag + np.array([0.0, 0.1], dtype=np.float64)
    normal_points = project_points(homography, np.vstack([segment_center_tag, normal_target_tag]))
    normal = normal_points[1] - normal_points[0]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        return []
    normal /= normal_norm

    polarity = 1.0 if segment.second_color > segment.first_color else -1.0
    edge_points: list[EdgePoint] = []
    offsets = np.arange(-profile_radius, profile_radius + 1, dtype=np.float64)

    for sample_index in range(samples_per_segment):
        alpha = (sample_index + 0.5) / samples_per_segment
        center = (1.0 - alpha) * p0 + alpha * p1
        values: list[float] = []

        for offset in offsets:
            value = bilinear_sample(gray, center + offset * normal)
            if value is None:
                values = []
                break
            values.append(value)

        if not values:
            continue

        profile = np.asarray(values, dtype=np.float64)
        profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 3), 0).reshape(-1)
        gradients = np.diff(profile)
        signed_gradients = polarity * gradients

        center_gradient_index = profile_radius - 1
        search_start = max(0, center_gradient_index - 2)
        search_end = min(len(signed_gradients), center_gradient_index + 3)
        if search_start >= search_end:
            continue

        local_index = int(np.argmax(signed_gradients[search_start:search_end]))
        gradient_index = search_start + local_index
        strength = float(signed_gradients[gradient_index])
        if strength < min_gradient:
            continue

        offset_position = float(offsets[gradient_index] + 0.5)
        if 0 < gradient_index < len(signed_gradients) - 1:
            offset_position += subpixel_peak_offset(
                signed_gradients[gradient_index - 1],
                signed_gradients[gradient_index],
                signed_gradients[gradient_index + 1],
            )

        edge_points.append(EdgePoint(point=center + offset_position * normal, weight=strength))

    return edge_points


def fit_weighted_line(points: list[EdgePoint], min_points: int) -> tuple[np.ndarray, float] | None:
    if len(points) < min_points:
        return None

    xy = np.vstack([point.point for point in points]).astype(np.float64)
    weights = np.asarray([point.weight for point in points], dtype=np.float64)
    weights = np.maximum(weights, 1e-6)
    centroid = np.average(xy, axis=0, weights=weights)
    centered = xy - centroid
    weighted = centered * np.sqrt(weights[:, None])
    _, _, vh = np.linalg.svd(weighted, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        return None
    normal /= normal_norm
    c = -float(normal @ centroid)
    line = np.array([normal[0], normal[1], c], dtype=np.float64)
    residuals = np.abs(xy @ normal + c)
    rms = float(np.sqrt(np.average(residuals * residuals, weights=weights)))
    return line, rms


def intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    point_h = np.cross(line_a, line_b)
    if abs(float(point_h[2])) < 1e-9:
        return None
    return point_h[:2] / point_h[2]


def grid_to_object_points(grid_points: np.ndarray, grid_size: int, marker_length: float) -> np.ndarray:
    xy = (grid_points / float(grid_size) - 0.5) * marker_length
    return np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)]).astype(np.float32)


def solve_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    initial_rvec: np.ndarray | None = None,
    initial_tvec: np.ndarray | None = None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None]:
    if len(object_points) < 4:
        return False, None, None, None

    image_points = image_points.astype(np.float32)
    if initial_rvec is None or initial_tvec is None:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return False, None, None, None
    else:
        rvec = np.asarray(initial_rvec, dtype=np.float64).reshape(3, 1).copy()
        tvec = np.asarray(initial_tvec, dtype=np.float64).reshape(3, 1).copy()

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        camera_matrix,
        None,
        rvec,
        tvec,
    )
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return True, rvec, tvec, float(np.sqrt(np.mean(errors * errors)))


def pose_rotation_delta_degrees(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    rotation_a = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))[0]
    rotation_b = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))[0]
    relative_rotation = rotation_a @ rotation_b.T
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pose_reprojection_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        axis=1,
    )
    return float(np.sqrt(np.mean(errors * errors)))


def validate_refined_pose(
    candidate_rvec: np.ndarray | None,
    candidate_tvec: np.ndarray | None,
    baseline_rvec: np.ndarray | None,
    baseline_tvec: np.ndarray | None,
    refined_corner_rms: float | None,
    *,
    max_translation_delta_m: float,
    max_translation_delta_depth_ratio: float,
    max_rotation_delta_deg: float,
    max_corner_reprojection_rms: float,
) -> tuple[bool, str | None, float | None, float | None]:
    if candidate_rvec is None or candidate_tvec is None:
        return False, "pose_solver_failed", None, None
    if baseline_rvec is None or baseline_tvec is None:
        return False, "baseline_pose_failed", None, None

    candidate_tvec = np.asarray(candidate_tvec, dtype=np.float64).reshape(3)
    baseline_tvec = np.asarray(baseline_tvec, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(candidate_tvec)):
        return False, "non_finite_translation", None, None
    if candidate_tvec[2] <= 0.0:
        return False, "non_positive_depth", None, None

    translation_delta = float(np.linalg.norm(candidate_tvec - baseline_tvec))
    translation_limit = max(
        max_translation_delta_m,
        max_translation_delta_depth_ratio * abs(float(baseline_tvec[2])),
    )
    if translation_delta > translation_limit:
        return False, "translation_delta", translation_delta, None

    rotation_delta = pose_rotation_delta_degrees(candidate_rvec, baseline_rvec)
    if rotation_delta > max_rotation_delta_deg:
        return False, "rotation_delta", translation_delta, rotation_delta

    if refined_corner_rms is None or not np.isfinite(refined_corner_rms):
        return False, "corner_reprojection_invalid", translation_delta, rotation_delta
    if refined_corner_rms > max_corner_reprojection_rms:
        return False, "corner_reprojection_rms", translation_delta, rotation_delta

    return True, None, translation_delta, rotation_delta


def rust_internal_line_refinement(
    gray: np.ndarray,
    cells: np.ndarray,
    homography: np.ndarray,
    samples_per_segment: int,
    profile_radius: int,
    min_gradient: float,
    min_line_points: int,
    max_line_rms: float,
) -> tuple[dict[tuple[str, int], list[EdgePoint]], dict[tuple[str, int], tuple[np.ndarray, float, int]], np.ndarray, np.ndarray] | None:
    if refine_internal_lines_rust is None:
        return None

    result = refine_internal_lines_rust(
        np.ascontiguousarray(gray),
        np.ascontiguousarray(cells),
        np.ascontiguousarray(homography, dtype=np.float64),
        samples_per_segment,
        profile_radius,
        min_gradient,
        min_line_points,
        max_line_rms,
    )
    line_records = np.asarray(result["line_records"], dtype=np.float64)
    object_grid = np.asarray(result["object_grid"], dtype=np.float64)
    image_xy = np.asarray(result["image_points"], dtype=np.float64)

    lines: dict[tuple[str, int], tuple[np.ndarray, float, int]] = {}
    for record in line_records:
        orientation = "vertical" if int(record[0]) == 0 else "horizontal"
        line_index = int(record[1])
        line = np.asarray(record[2:5], dtype=np.float64)
        rms = float(record[5])
        count = int(record[6])
        lines[(orientation, line_index)] = (line, rms, count)

    return {}, lines, object_grid, image_xy


def python_internal_line_refinement(
    gray: np.ndarray,
    cells: np.ndarray,
    homography: np.ndarray,
    samples_per_segment: int,
    profile_radius: int,
    min_gradient: float,
    min_line_points: int,
    max_line_rms: float,
) -> tuple[dict[tuple[str, int], list[EdgePoint]], dict[tuple[str, int], tuple[np.ndarray, float, int]], np.ndarray, np.ndarray]:
    grid_size = int(cells.shape[0])
    buckets: dict[tuple[str, int], list[EdgePoint]] = {}
    for segment in visible_edge_segments(cells):
        edge_points = detect_edge_points_for_segment(
            gray,
            homography,
            segment,
            samples_per_segment=samples_per_segment,
            profile_radius=profile_radius,
            min_gradient=min_gradient,
        )
        if edge_points:
            buckets.setdefault((segment.orientation, segment.line_index), []).extend(edge_points)

    lines: dict[tuple[str, int], tuple[np.ndarray, float, int]] = {}
    for key, points in buckets.items():
        fit = fit_weighted_line(points, min_points=min_line_points)
        if fit is None:
            continue
        line, rms = fit
        if rms <= max_line_rms:
            lines[key] = (line, rms, len(points))

    object_grid_points: list[list[float]] = []
    image_points: list[np.ndarray] = []
    for col in range(1, grid_size):
        vertical = lines.get(("vertical", col))
        if vertical is None:
            continue
        for row in range(1, grid_size):
            horizontal = lines.get(("horizontal", row))
            if horizontal is None:
                continue
            image_point = intersect_lines(vertical[0], horizontal[0])
            if image_point is None:
                continue
            object_grid_points.append([float(col), float(row)])
            image_points.append(image_point)

    return (
        buckets,
        lines,
        np.asarray(object_grid_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
    )


def refine_tag_from_internal_lines(
    gray: np.ndarray,
    corners: np.ndarray,
    marker_id: int,
    dictionary: cv2.aruco.Dictionary,
    camera_matrix: np.ndarray,
    marker_length: float,
    samples_per_segment: int,
    profile_radius: int,
    min_gradient: float,
    min_line_points: int,
    max_line_rms: float,
    max_translation_delta_m: float = 0.02,
    max_translation_delta_depth_ratio: float = 0.10,
    max_rotation_delta_deg: float = 10.0,
    max_corner_reprojection_rms: float = 1.5,
) -> dict[str, object]:
    cells = build_marker_cell_grid(dictionary, marker_id)
    grid_size = int(cells.shape[0])
    tag_corners = np.array(
        [[0.0, 0.0], [grid_size, 0.0], [grid_size, grid_size], [0.0, grid_size]],
        dtype=np.float64,
    )
    homography, _ = cv2.findHomography(tag_corners, corners.astype(np.float64))

    internal_result = rust_internal_line_refinement(
        gray,
        cells,
        homography,
        samples_per_segment=samples_per_segment,
        profile_radius=profile_radius,
        min_gradient=min_gradient,
        min_line_points=min_line_points,
        max_line_rms=max_line_rms,
    )
    backend = "rust"
    if internal_result is None:
        internal_result = python_internal_line_refinement(
            gray,
            cells,
            homography,
            samples_per_segment=samples_per_segment,
            profile_radius=profile_radius,
            min_gradient=min_gradient,
            min_line_points=min_line_points,
            max_line_rms=max_line_rms,
        )
        backend = "python"

    corner_object_grid = np.asarray([[0.0, 0.0], [grid_size, 0.0], [grid_size, grid_size], [0.0, grid_size]])
    corner_object = grid_to_object_points(corner_object_grid, grid_size, marker_length)
    corner_ok, corner_rvec, corner_tvec, corner_rms = solve_pose(
        corner_object,
        corners.astype(np.float64),
        camera_matrix,
    )

    buckets, lines, object_grid, image_xy = internal_result
    object_points = grid_to_object_points(object_grid, grid_size, marker_length)
    candidate_ok, candidate_rvec, candidate_tvec, candidate_reprojection_rms = solve_pose(
        object_points,
        image_xy,
        camera_matrix,
        initial_rvec=corner_rvec if corner_ok else None,
        initial_tvec=corner_tvec if corner_ok else None,
    )
    refined_corner_rms = None
    if candidate_ok and candidate_rvec is not None and candidate_tvec is not None:
        refined_corner_rms = pose_reprojection_rms(
            corner_object,
            corners,
            camera_matrix,
            candidate_rvec,
            candidate_tvec,
        )

    refinement_accepted, rejection_reason, translation_delta, rotation_delta = validate_refined_pose(
        candidate_rvec if candidate_ok else None,
        candidate_tvec if candidate_ok else None,
        corner_rvec if corner_ok else None,
        corner_tvec if corner_ok else None,
        refined_corner_rms,
        max_translation_delta_m=max_translation_delta_m,
        max_translation_delta_depth_ratio=max_translation_delta_depth_ratio,
        max_rotation_delta_deg=max_rotation_delta_deg,
        max_corner_reprojection_rms=max_corner_reprojection_rms,
    )
    if refinement_accepted:
        pose_ok = True
        rvec = candidate_rvec
        tvec = candidate_tvec
        reprojection_rms = candidate_reprojection_rms
        pose_source = "refined"
    else:
        pose_ok = corner_ok
        rvec = corner_rvec
        tvec = corner_tvec
        reprojection_rms = corner_rms
        pose_source = "baseline" if corner_ok else "none"

    return {
        "cells": cells,
        "buckets": buckets,
        "lines": lines,
        "object_grid": object_grid,
        "image_points": image_xy,
        "pose_ok": pose_ok,
        "rvec": rvec,
        "tvec": tvec,
        "reprojection_rms": reprojection_rms,
        "pose_source": pose_source,
        "refinement_accepted": refinement_accepted,
        "rejection_reason": rejection_reason,
        "candidate_pose_ok": candidate_ok,
        "candidate_rvec": candidate_rvec,
        "candidate_tvec": candidate_tvec,
        "candidate_reprojection_rms": candidate_reprojection_rms,
        "refined_corner_reprojection_rms": refined_corner_rms,
        "translation_delta_m": translation_delta,
        "rotation_delta_deg": rotation_delta,
        "corner_pose_ok": corner_ok,
        "corner_rvec": corner_rvec,
        "corner_tvec": corner_tvec,
        "corner_reprojection_rms": corner_rms,
        "backend": backend,
    }


def draw_debug(
    gray: np.ndarray,
    corners: np.ndarray,
    marker_id: int,
    result: dict[str, object],
    output_path: Path,
) -> None:
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.polylines(debug, [corners.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.putText(
        debug,
        f"id={marker_id}",
        tuple(corners[0].astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    lines = result["lines"]
    assert isinstance(lines, dict)
    height, width = gray.shape[:2]
    for (orientation, _), (line, _rms, _count) in lines.items():
        color = (0, 180, 255) if orientation == "vertical" else (255, 120, 0)
        a, b, c = line
        endpoints: list[tuple[int, int]] = []
        if abs(b) > abs(a):
            for x in [0, width - 1]:
                y = -(a * x + c) / b
                endpoints.append((int(round(x)), int(round(y))))
        else:
            for y in [0, height - 1]:
                x = -(b * y + c) / a
                endpoints.append((int(round(x)), int(round(y))))
        cv2.line(debug, endpoints[0], endpoints[1], color, 1, cv2.LINE_AA)

    image_points = result["image_points"]
    if isinstance(image_points, np.ndarray):
        for point in image_points:
            cv2.circle(debug, tuple(np.round(point).astype(int)), 3, (0, 255, 0), -1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/3marker_april_mono_160fov_3/webcam_color.msgpack"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration/good.toml"))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--tag-index", type=int, default=0)
    parser.add_argument("--samples-per-segment", type=int, default=5)
    parser.add_argument("--profile-radius", type=int, default=5)
    parser.add_argument("--min-gradient", type=float, default=8.0)
    parser.add_argument("--min-line-points", type=int, default=5)
    parser.add_argument("--max-line-rms", type=float, default=2.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/line_refine_debug.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    frame = unpack_frame(args.data, args.frame_index)
    gray, camera_matrix = undistort_frame(frame, calibration)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    corners_list, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No DICT_APRILTAG_36h11 markers detected in the selected frame")
    if args.tag_index < 0 or args.tag_index >= len(ids):
        raise IndexError(f"tag-index must be in [0, {len(ids) - 1}]")

    marker_id = int(ids[args.tag_index, 0])
    corners = corners_list[args.tag_index].reshape(4, 2)
    result = refine_tag_from_internal_lines(
        gray,
        corners,
        marker_id,
        dictionary,
        camera_matrix,
        calibration.marker_length,
        samples_per_segment=args.samples_per_segment,
        profile_radius=args.profile_radius,
        min_gradient=args.min_gradient,
        min_line_points=args.min_line_points,
        max_line_rms=args.max_line_rms,
    )
    draw_debug(gray, corners, marker_id, result, args.output)

    lines = result["lines"]
    image_points = result["image_points"]
    print(f"frame_index={args.frame_index}")
    print(f"detected_ids={ids.ravel().tolist()}")
    print(f"selected_id={marker_id}")
    print(f"line_count={len(lines) if isinstance(lines, dict) else 0}")
    print(f"intersection_count={len(image_points) if isinstance(image_points, np.ndarray) else 0}")
    print(f"corner_pose_ok={result['corner_pose_ok']}")
    print(f"corner_reprojection_rms_px={result['corner_reprojection_rms']}")
    print(f"refined_pose_ok={result['pose_ok']}")
    print(f"refined_reprojection_rms_px={result['reprojection_rms']}")
    print(f"backend={result['backend']}")
    print(f"debug_image={args.output}")

    if result["pose_ok"]:
        rvec = result["rvec"]
        tvec = result["tvec"]
        assert isinstance(rvec, np.ndarray)
        assert isinstance(tvec, np.ndarray)
        print(f"rvec={rvec.ravel().tolist()}")
        print(f"tvec_m={tvec.ravel().tolist()}")


if __name__ == "__main__":
    main()
