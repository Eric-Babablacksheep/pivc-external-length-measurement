from __future__ import annotations

import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from session_store import BaselineSessionStore


def measured(length_cm: float) -> dict:
    return {
        "measurement_status": "MEASURED",
        "external_length_cm": length_cm,
        "diagnostic_url": f"/diagnostics/{length_cm}.jpg",
    }


def settings() -> dict:
    return {
        "confidence": 0.50,
        "iou": 0.70,
        "tolerance_cm": 0.10,
        "imgsz": 960,
    }


def indicator(code: str = "WITHIN_TOLERANCE") -> dict:
    return {
        "code": code,
        "label": "Within measurement tolerance",
    }


class BaselineSessionStoreFollowUpTests(unittest.TestCase):
    def setUp(self):
        self.store = BaselineSessionStore()
        self.session = self.store.create(measured(3.5), settings())

    def test_new_session_starts_with_empty_follow_up_history(self):
        self.assertEqual([], self.session["follow_ups"])

    def test_append_follow_up_calculates_signed_and_absolute_change(self):
        updated = self.store.add_follow_up(
            self.session["session_id"],
            measured(3.32),
            indicator("INWARD_DECREASE"),
        )

        follow_up = updated["follow_ups"][0]
        self.assertEqual("FOLLOW_UP_MEASURED", updated["session_status"])
        self.assertEqual(-0.18, follow_up["signed_change_cm"])
        self.assertEqual(0.18, follow_up["absolute_change_cm"])
        self.assertEqual(3.5, updated["baseline"]["external_length_cm"])

    def test_multiple_follow_ups_are_preserved_in_order(self):
        self.store.add_follow_up(
            self.session["session_id"],
            measured(3.4),
            indicator(),
        )
        updated = self.store.add_follow_up(
            self.session["session_id"],
            measured(3.25),
            indicator("INWARD_DECREASE"),
        )

        self.assertEqual(2, len(updated["follow_ups"]))
        self.assertEqual(3.4, updated["follow_ups"][0]["measurement"]["external_length_cm"])
        self.assertEqual(3.25, updated["follow_ups"][1]["measurement"]["external_length_cm"])

    def test_rejected_measurement_cannot_enter_successful_history(self):
        with self.assertRaisesRegex(ValueError, "successful measurement"):
            self.store.add_follow_up(
                self.session["session_id"],
                {"measurement_status": "REJECTED"},
                indicator(),
            )

        unchanged = self.store.get(self.session["session_id"])
        self.assertEqual([], unchanged["follow_ups"])

    def test_unknown_session_returns_none(self):
        self.assertIsNone(
            self.store.add_follow_up(
                "missing",
                measured(3.4),
                indicator(),
            )
        )

    def test_session_preserves_settings_snapshot(self):
        source_settings = settings()

        session = self.store.create(measured(3.5), source_settings)
        source_settings["confidence"] = 0.90

        self.assertEqual(0.50, session["settings"]["confidence"])

    def test_follow_up_preserves_settings_and_indicator(self):
        research_indicator = indicator("OUTWARD_INCREASE")

        updated = self.store.add_follow_up(
            self.session["session_id"],
            measured(3.7),
            research_indicator,
        )
        research_indicator["code"] = "MUTATED"

        self.assertEqual(settings(), updated["settings"])
        self.assertEqual(
            "OUTWARD_INCREASE",
            updated["follow_ups"][0]["research_indicator"]["code"],
        )


if __name__ == "__main__":
    unittest.main()
