import unittest

import cv2
import numpy as np

from scripts.line_refine_single_frame import (
    refine_outer_corners_subpix,
    solve_square_pose,
    validate_refined_pose,
)


def validate(
    candidate_rvec: np.ndarray,
    candidate_tvec: np.ndarray,
    baseline_rvec: np.ndarray | None = None,
    baseline_tvec: np.ndarray | None = None,
    corner_rms: float = 0.4,
) -> tuple[bool, str | None, float | None, float | None]:
    if baseline_rvec is None:
        baseline_rvec = np.zeros(3)
    if baseline_tvec is None:
        baseline_tvec = np.array([0.0, 0.0, 0.5])
    return validate_refined_pose(
        candidate_rvec,
        candidate_tvec,
        baseline_rvec,
        baseline_tvec,
        corner_rms,
        max_translation_delta_m=0.02,
        max_translation_delta_depth_ratio=0.10,
        max_rotation_delta_deg=10.0,
        max_corner_reprojection_rms=1.5,
    )


class RefinedPoseAcceptanceTests(unittest.TestCase):
    def test_accepts_small_pose_correction(self) -> None:
        accepted, reason, translation_delta, rotation_delta = validate(
            np.array([0.01, -0.02, 0.01]),
            np.array([0.002, -0.003, 0.505]),
        )

        self.assertTrue(accepted)
        self.assertIsNone(reason)
        self.assertIsNotNone(translation_delta)
        self.assertLess(translation_delta, 0.01)
        self.assertIsNotNone(rotation_delta)
        self.assertLess(rotation_delta, 2.0)

    def test_rejects_large_translation_jump(self) -> None:
        accepted, reason, translation_delta, _ = validate(
            np.zeros(3),
            np.array([0.4, -0.3, 0.9]),
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "translation_delta")
        self.assertIsNotNone(translation_delta)
        self.assertGreater(translation_delta, 0.5)

    def test_rejects_large_rotation_jump(self) -> None:
        accepted, reason, _, rotation_delta = validate(
            np.array([0.0, 0.0, np.deg2rad(15.0)]),
            np.array([0.0, 0.0, 0.5]),
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "rotation_delta")
        self.assertIsNotNone(rotation_delta)
        self.assertGreater(rotation_delta, 10.0)

    def test_rejects_bad_outer_corner_reprojection(self) -> None:
        accepted, reason, _, _ = validate(
            np.zeros(3),
            np.array([0.0, 0.0, 0.5]),
            corner_rms=2.0,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "corner_reprojection_rms")


class OuterCornerCandidateTests(unittest.TestCase):
    def test_subpixel_refinement_preserves_shape_and_input(self) -> None:
        gray = np.zeros((80, 80), dtype=np.uint8)
        gray[20:60, 20:60] = 255
        corners = np.array(
            [[19.5, 19.5], [59.5, 19.5], [59.5, 59.5], [19.5, 59.5]],
            dtype=np.float64,
        )
        original = corners.copy()

        refined = refine_outer_corners_subpix(gray, corners, window_radius=4)

        self.assertEqual(refined.shape, (4, 2))
        self.assertTrue(np.isfinite(refined).all())
        np.testing.assert_array_equal(corners, original)

    def test_ippe_square_recovers_synthetic_pose(self) -> None:
        camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        marker_length = 0.05
        half = marker_length * 0.5
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        expected_rvec = np.array([0.15, -0.10, 0.05], dtype=np.float64)
        expected_tvec = np.array([0.01, -0.02, 0.6], dtype=np.float64)
        corners, _ = cv2.projectPoints(
            object_points,
            expected_rvec,
            expected_tvec,
            camera_matrix,
            None,
        )

        ok, rvec, tvec, rms = solve_square_pose(
            corners.reshape(4, 2),
            camera_matrix,
            marker_length,
        )

        self.assertTrue(ok)
        self.assertIsNotNone(rvec)
        self.assertIsNotNone(tvec)
        self.assertIsNotNone(rms)
        np.testing.assert_allclose(tvec.reshape(3), expected_tvec, atol=1e-5)
        self.assertLess(rms, 1e-4)

    def test_ippe_square_uses_reference_to_disambiguate_pose(self) -> None:
        camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        marker_length = 0.05
        half = marker_length * 0.5
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        expected_rvec = np.array([0.02, -0.03, 0.01], dtype=np.float64)
        expected_tvec = np.array([0.0, 0.0, 0.8], dtype=np.float64)
        corners, _ = cv2.projectPoints(
            object_points,
            expected_rvec,
            expected_tvec,
            camera_matrix,
            None,
        )

        ok, rvec, tvec, _rms = solve_square_pose(
            corners.reshape(4, 2),
            camera_matrix,
            marker_length,
            reference_rvec=expected_rvec,
            reference_tvec=expected_tvec,
        )

        self.assertTrue(ok)
        self.assertIsNotNone(rvec)
        self.assertIsNotNone(tvec)
        np.testing.assert_allclose(tvec.reshape(3), expected_tvec, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
