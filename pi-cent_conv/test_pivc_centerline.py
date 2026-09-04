import unittest

import numpy as np

from pivc_centerline import (
    calibrate_length_from_marks,
    reconstruct_continuous_centerline,
    repair_centerline_through_marks,
)


def component(component_id, start, end, width, samples=30):
    path = np.linspace(start, end, samples, dtype=np.float64)
    return {
        "component_id": component_id,
        "path_array": path,
        "median_width": float(width),
    }


class ContinuousCenterlineTests(unittest.TestCase):
    def test_joins_aligned_sections_and_skips_narrow_fragment(self):
        components = [
            component(0, (10, 50), (100, 50), 36),
            component(1, (108, 50), (160, 50), 40),
            component(2, (165, 56), (170, 56), 10),
            component(3, (177, 50), (240, 50), 40),
        ]

        result = reconstruct_continuous_centerline(components)

        self.assertEqual(result["component_ids"], [0, 1, 3])
        self.assertEqual(len(result["bridges"]), 2)
        self.assertTrue(result["is_continuous"])
        step_lengths = np.linalg.norm(
            np.diff(result["path"], axis=0), axis=1
        )
        self.assertLessEqual(float(step_lengths.max()), 2.0)

    def test_rejects_gap_that_is_too_large(self):
        components = [
            component(0, (10, 50), (100, 50), 40),
            component(1, (250, 50), (320, 50), 40),
        ]

        with self.assertRaisesRegex(RuntimeError, "unsafe gap"):
            reconstruct_continuous_centerline(components)

    def test_preserves_a_gently_curved_ordered_path(self):
        components = [
            component(0, (10, 40), (80, 45), 35),
            component(1, (88, 46), (150, 55), 36),
            component(2, (158, 57), (220, 70), 34),
        ]

        result = reconstruct_continuous_centerline(components)

        self.assertEqual(result["component_ids"], [0, 1, 2])
        self.assertEqual(tuple(result["path"][0]), (10.0, 40.0))
        self.assertEqual(tuple(result["path"][-1]), (220.0, 70.0))
        self.assertGreater(result["total_length_px"], 200.0)

    def test_builds_horizontal_chain_despite_local_endpoint_distortion(self):
        left = np.vstack([
            np.column_stack([
                np.full(81, 50.0), np.arange(0.0, 81.0)
            ]),
            np.column_stack([
                np.arange(31.0, 51.0), np.full(20, 100.0)
            ]),
        ])
        middle = np.vstack([
            np.column_stack([
                np.arange(50.0, 30.0, -1.0), np.full(20, 110.0)
            ]),
            np.column_stack([
                np.full(81, 50.0), np.arange(110.0, 191.0)
            ]),
        ])
        right = np.column_stack([
            np.full(61, 50.0), np.arange(200.0, 261.0)
        ])
        components = [
            {"component_id": 1, "path_array": middle, "median_width": 40.0},
            {"component_id": 2, "path_array": right, "median_width": 39.0},
            {"component_id": 3, "path_array": left, "median_width": 41.0},
        ]

        result = reconstruct_continuous_centerline(components)

        self.assertIn(result["component_ids"], ([3, 1, 2], [2, 1, 3]))
        self.assertEqual(len(result["bridges"]), 2)
        self.assertTrue(result["is_continuous"])


class MarkAwareRepairTests(unittest.TestCase):
    def test_repairs_connected_sideways_detour_through_mark_centre(self):
        rows = np.arange(0.0, 101.0)
        columns = np.zeros_like(rows)
        detour = np.abs(rows - 50.0) <= 8.0
        columns[detour] = 10.0 * (1.0 - np.abs(rows[detour] - 50.0) / 9.0)
        path = np.column_stack([rows, columns])

        result = repair_centerline_through_marks(
            path, mark_centres=[(50.0, 0.0)], pivc_width=10.0
        )

        distance_to_mark = np.linalg.norm(
            result["path"] - np.array([50.0, 0.0]), axis=1
        ).min()
        self.assertLess(distance_to_mark, 0.25)
        self.assertEqual(result["repaired_mark_indices"], [0])
        self.assertLess(result["total_length_px"], 102.0)
        np.testing.assert_allclose(result["path"][0], path[0])
        np.testing.assert_allclose(result["path"][-1], path[-1])


class PixelToCentimetreCalibrationTests(unittest.TestCase):
    def test_uses_all_consecutive_one_centimetre_mark_intervals(self):
        path = np.column_stack([
            np.arange(0.0, 401.0), np.zeros(401, dtype=np.float64)
        ])
        marks = [(100.0, 0.0), (200.0, 0.0), (300.0, 0.0)]

        result = calibrate_length_from_marks(
            path, marks, pivc_width=20.0, known_spacing_cm=1.0
        )

        np.testing.assert_allclose(result["spacing_pixels"], [100.0, 100.0])
        self.assertAlmostEqual(result["pixels_per_cm"], 100.0)
        self.assertAlmostEqual(result["length_cm"], 4.0)

    def test_uses_centreline_arc_distance_for_a_curved_pivc(self):
        path = np.vstack([
            np.column_stack([
                np.zeros(101, dtype=np.float64), np.arange(0.0, 101.0)
            ]),
            np.column_stack([
                np.arange(1.0, 101.0), np.full(100, 100.0)
            ]),
        ])
        marks = [(0.0, 50.0), (0.0, 100.0), (50.0, 100.0)]

        result = calibrate_length_from_marks(
            path, marks, pivc_width=10.0, known_spacing_cm=1.0
        )

        np.testing.assert_allclose(result["spacing_pixels"], [50.0, 50.0])
        self.assertAlmostEqual(result["length_cm"], 4.0)

    def test_rejects_inconsistent_consecutive_mark_spacing(self):
        path = np.column_stack([
            np.arange(0.0, 401.0), np.zeros(401, dtype=np.float64)
        ])

        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            calibrate_length_from_marks(
                path,
                [(100.0, 0.0), (200.0, 0.0), (350.0, 0.0)],
                pivc_width=20.0,
            )

    def test_rejects_when_fewer_than_two_marks_are_available(self):
        path = np.column_stack([
            np.arange(0.0, 101.0), np.zeros(101, dtype=np.float64)
        ])

        with self.assertRaisesRegex(RuntimeError, "at least two"):
            calibrate_length_from_marks(path, [(50.0, 0.0)], pivc_width=10.0)

    def test_rejects_a_mark_far_from_the_pivc_centreline(self):
        path = np.column_stack([
            np.arange(0.0, 201.0), np.zeros(201, dtype=np.float64)
        ])

        with self.assertRaisesRegex(RuntimeError, "too far"):
            calibrate_length_from_marks(
                path,
                [(50.0, 0.0), (150.0, 100.0)],
                pivc_width=10.0,
            )

    def test_leaves_path_unchanged_when_mark_is_distant(self):
        path = np.column_stack(
            [np.arange(0.0, 51.0), np.zeros(51, dtype=np.float64)]
        )

        result = repair_centerline_through_marks(
            path, mark_centres=[(25.0, 100.0)], pivc_width=10.0
        )

        self.assertEqual(result["repaired_mark_indices"], [])
        np.testing.assert_allclose(result["path"], path)

    def test_repairs_multiple_marks_without_changing_endpoints(self):
        rows = np.arange(0.0, 151.0)
        columns = 0.01 * rows
        for centre_row in (50.0, 100.0):
            affected = np.abs(rows - centre_row) <= 6.0
            columns[affected] += 6.0 * (
                1.0 - np.abs(rows[affected] - centre_row) / 7.0
            )
        path = np.column_stack([rows, columns])
        marks = [(50.0, 0.5), (100.0, 1.0)]

        result = repair_centerline_through_marks(
            path, mark_centres=marks, pivc_width=10.0
        )

        self.assertEqual(result["repaired_mark_indices"], [0, 1])
        for mark in marks:
            self.assertLess(
                np.linalg.norm(result["path"] - np.array(mark), axis=1).min(),
                0.25,
            )
        np.testing.assert_allclose(result["path"][0], path[0])
        np.testing.assert_allclose(result["path"][-1], path[-1])


if __name__ == "__main__":
    unittest.main()
