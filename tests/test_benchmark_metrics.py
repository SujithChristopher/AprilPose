import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from aprilpose.benchmark.metrics import (
    orientation_change_summary,
    timing_summary,
    translation_summary,
)


class BenchmarkMetricTests(unittest.TestCase):
    def test_translation_summary_aligns_rigid_coordinate_frames(self) -> None:
        reference = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.2, 0.0], [0.2, 0.2, 0.1]]
        )
        rotation = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        predicted = (rotation @ reference.T).T + np.array([1.0, -2.0, 0.5])

        summary = translation_summary(predicted, reference)

        self.assertEqual(summary["sample_count"], 4)
        self.assertLess(summary["rmse_m"], 1e-10)

    def test_orientation_change_summary_is_frame_invariant(self) -> None:
        reference = Rotation.from_euler(
            "z",
            [[0.0], [2.0], [5.0]],
            degrees=True,
        )
        fixed_offset = Rotation.from_euler("x", 90.0, degrees=True)
        tag = fixed_offset * reference
        rvecs = [rotation.as_rotvec() for rotation in tag]

        summary = orientation_change_summary(rvecs, reference)

        self.assertEqual(summary["sample_count"], 2)
        self.assertLess(summary["rmse_deg"], 1e-10)

    def test_timing_summary(self) -> None:
        summary = timing_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["median_ms"], 2.0)


if __name__ == "__main__":
    unittest.main()
