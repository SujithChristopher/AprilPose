from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from cv2 import aruco

from scripts.line_refine_single_frame import (
    grid_to_object_points,
    refine_pose_from_outer_corners_subpix,
    refine_tag_from_internal_lines,
    solve_pose,
)

from .config import BenchmarkConfig


@dataclass
class PoseResult:
    ok: bool
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    reprojection_rms_px: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


class PoseModel(Protocol):
    name: str

    def reset(self) -> None: ...

    def predict(
        self,
        gray: np.ndarray,
        corners: np.ndarray,
        marker_id: int,
        dictionary: aruco.Dictionary,
        camera_matrix: np.ndarray,
        marker_length: float,
    ) -> PoseResult: ...


class BaselineModel:
    name = "baseline"

    def reset(self) -> None:
        return None

    def predict(
        self,
        gray: np.ndarray,
        corners: np.ndarray,
        marker_id: int,
        dictionary: aruco.Dictionary,
        camera_matrix: np.ndarray,
        marker_length: float,
    ) -> PoseResult:
        del gray, marker_id, dictionary
        object_grid = np.asarray(
            [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]],
            dtype=np.float64,
        )
        object_points = grid_to_object_points(
            object_grid,
            grid_size=8,
            marker_length=marker_length,
        )
        ok, rvec, tvec, rms = solve_pose(
            object_points,
            corners.astype(np.float64),
            camera_matrix,
        )
        return PoseResult(ok, rvec, tvec, rms)


class SubpixelIppeTemporalModel:
    name = "subpix_ippe_temporal"

    def __init__(self, window_radius: int = 5) -> None:
        self.window_radius = window_radius
        self.reset()

    def reset(self) -> None:
        self.previous_rvec: np.ndarray | None = None
        self.previous_tvec: np.ndarray | None = None

    def predict(
        self,
        gray: np.ndarray,
        corners: np.ndarray,
        marker_id: int,
        dictionary: aruco.Dictionary,
        camera_matrix: np.ndarray,
        marker_length: float,
    ) -> PoseResult:
        del marker_id, dictionary
        result = refine_pose_from_outer_corners_subpix(
            gray,
            corners,
            camera_matrix,
            marker_length,
            window_radius=self.window_radius,
            reference_rvec=self.previous_rvec,
            reference_tvec=self.previous_tvec,
        )
        if result["pose_ok"]:
            self.previous_rvec = result["rvec"]
            self.previous_tvec = result["tvec"]
        else:
            self.reset()
        return PoseResult(
            bool(result["pose_ok"]),
            result["rvec"],
            result["tvec"],
            result["reprojection_rms"],
        )


class InternalLinesModel:
    name = "internal_lines"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def reset(self) -> None:
        return None

    def predict(
        self,
        gray: np.ndarray,
        corners: np.ndarray,
        marker_id: int,
        dictionary: aruco.Dictionary,
        camera_matrix: np.ndarray,
        marker_length: float,
    ) -> PoseResult:
        result = refine_tag_from_internal_lines(
            gray,
            corners,
            marker_id,
            dictionary,
            camera_matrix,
            marker_length,
            samples_per_segment=self.config.samples_per_segment,
            profile_radius=self.config.profile_radius,
            min_gradient=self.config.min_gradient,
            min_line_points=self.config.min_line_points,
            max_line_rms=self.config.max_line_rms,
        )
        image_points = result["image_points"]
        lines = result["lines"]
        return PoseResult(
            bool(result["pose_ok"]),
            result["rvec"],
            result["tvec"],
            result["reprojection_rms"],
            {
                "refinement_accepted": bool(result["refinement_accepted"]),
                "rejection_reason": result["rejection_reason"],
                "pose_source": result["pose_source"],
                "line_count": len(lines) if isinstance(lines, dict) else 0,
                "intersection_count": (
                    len(image_points) if isinstance(image_points, np.ndarray) else 0
                ),
            },
        )


def build_models(config: BenchmarkConfig) -> list[PoseModel]:
    factories = {
        "baseline": BaselineModel,
        "subpix_ippe_temporal": SubpixelIppeTemporalModel,
        "internal_lines": lambda: InternalLinesModel(config),
    }
    unknown = [name for name in config.model_names if name not in factories]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Available: {', '.join(factories)}"
        )
    return [factories[name]() for name in config.model_names]

