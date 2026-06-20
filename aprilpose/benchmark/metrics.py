from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rigid_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    u, _singular, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return (rotation @ source.T).T + translation


def translation_summary(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float | int | None]:
    valid = np.isfinite(predicted).all(axis=1) & np.isfinite(reference).all(axis=1)
    if np.count_nonzero(valid) < 3:
        return {"sample_count": int(np.count_nonzero(valid)), "mean_m": None, "rmse_m": None, "p95_m": None}
    pred = predicted[valid] - predicted[valid][0]
    ref = reference[valid]
    ref = rigid_align(ref, pred)
    errors = np.linalg.norm(pred - ref, axis=1)
    return {
        "sample_count": len(errors),
        "mean_m": float(np.mean(errors)),
        "rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "p95_m": float(np.percentile(errors, 95)),
    }


def orientation_change_summary(
    rvecs: list[np.ndarray | None],
    reference_rotations: Rotation,
) -> dict[str, float | int | None]:
    signed_errors: list[float] = []
    previous_tag: Rotation | None = None
    previous_reference: Rotation | None = None
    for rvec, reference in zip(rvecs, reference_rotations, strict=True):
        if rvec is None or not np.isfinite(rvec).all():
            previous_tag = None
            previous_reference = None
            continue
        tag = Rotation.from_rotvec(rvec.reshape(3))
        if previous_tag is not None and previous_reference is not None:
            tag_delta = np.degrees((previous_tag.inv() * tag).magnitude())
            reference_delta = np.degrees(
                (previous_reference.inv() * reference).magnitude()
            )
            signed_errors.append(float(tag_delta - reference_delta))
        previous_tag = tag
        previous_reference = reference
    if not signed_errors:
        return {"sample_count": 0, "mean_abs_deg": None, "median_abs_deg": None, "p95_abs_deg": None, "rmse_deg": None}
    errors = np.asarray(signed_errors, dtype=np.float64)
    absolute = np.abs(errors)
    return {
        "sample_count": len(errors),
        "mean_abs_deg": float(np.mean(absolute)),
        "median_abs_deg": float(np.median(absolute)),
        "p95_abs_deg": float(np.percentile(absolute, 95)),
        "rmse_deg": float(np.sqrt(np.mean(errors * errors))),
    }


def timing_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"sample_count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(array),
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }

