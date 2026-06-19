import unittest

import numpy as np

from scripts.line_refine_single_frame import validate_refined_pose


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


if __name__ == "__main__":
    unittest.main()
