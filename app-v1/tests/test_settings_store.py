from __future__ import annotations

import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from settings_store import ResearchSettingsStore


class ResearchSettingsStoreTests(unittest.TestCase):
    def test_defaults_are_validated_values(self):
        settings = ResearchSettingsStore().get()

        self.assertEqual(0.50, settings.confidence)
        self.assertEqual(0.70, settings.iou)
        self.assertEqual(0.10, settings.tolerance_cm)
        self.assertEqual(960, settings.imgsz)

    def test_update_returns_immutable_snapshot(self):
        store = ResearchSettingsStore()
        snapshot = store.update(
            confidence=0.35,
            iou=0.60,
            tolerance_cm=0.15,
        )

        store.reset()

        self.assertEqual(0.35, snapshot.confidence)
        self.assertEqual(0.50, store.get().confidence)

    def test_boundaries_are_enforced(self):
        store = ResearchSettingsStore()
        invalid_values = {
            "confidence": (0.04, 0.96),
            "iou": (0.09, 0.96),
            "tolerance_cm": (0.0, 1.01),
        }

        for field, values in invalid_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    kwargs = {
                        "confidence": 0.5,
                        "iou": 0.7,
                        "tolerance_cm": 0.1,
                    }
                    kwargs[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        store.update(**kwargs)

    def test_boundary_values_are_accepted(self):
        store = ResearchSettingsStore()

        low = store.update(
            confidence=0.05,
            iou=0.10,
            tolerance_cm=0.01,
        )
        high = store.update(
            confidence=0.95,
            iou=0.95,
            tolerance_cm=1.00,
        )

        self.assertEqual((0.05, 0.10, 0.01), (
            low.confidence,
            low.iou,
            low.tolerance_cm,
        ))
        self.assertEqual((0.95, 0.95, 1.00), (
            high.confidence,
            high.iou,
            high.tolerance_cm,
        ))


if __name__ == "__main__":
    unittest.main()
