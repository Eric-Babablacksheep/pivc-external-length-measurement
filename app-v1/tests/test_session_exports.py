from __future__ import annotations

import csv
import sys
import unittest
from io import StringIO
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from session_exports import session_csv, session_export_payload


def example_session() -> dict:
    return {
        "session_id": "session-123",
        "session_status": "FOLLOW_UP_MEASURED",
        "created_at": "2026-08-30T10:00:00+00:00",
        "settings": {
            "confidence": 0.5,
            "iou": 0.7,
            "tolerance_cm": 0.1,
            "imgsz": 960,
        },
        "baseline": {
            "measurement_status": "MEASURED",
            "external_length_cm": 3.5,
        },
        "follow_ups": [
            {
                "follow_up_id": "follow-1",
                "created_at": "2026-08-30T10:05:00+00:00",
                "measurement": {
                    "measurement_status": "MEASURED",
                    "external_length_cm": 3.72,
                },
                "signed_change_cm": 0.22,
                "absolute_change_cm": 0.22,
                "research_indicator": {
                    "code": "OUTWARD_INCREASE",
                },
            }
        ],
    }


class SessionExportTests(unittest.TestCase):
    def test_session_list_and_export_routes_are_declared(self):
        source = (APP_DIR / "main.py").read_text(encoding="utf-8")

        self.assertIn('@app.get("/api/v1/sessions")', source)
        self.assertIn(
            '@app.get("/api/v1/sessions/{session_id}/export.json")',
            source,
        )
        self.assertIn(
            '@app.get("/api/v1/sessions/{session_id}/export.csv")',
            source,
        )

    def test_json_payload_is_independent_and_contains_warning(self):
        session = example_session()

        payload = session_export_payload(session)
        payload["baseline"]["external_length_cm"] = 99

        self.assertEqual(3.5, session["baseline"]["external_length_cm"])
        self.assertIn("not clinically validated", payload["warning"])
        self.assertEqual(1, len(payload["follow_ups"]))

    def test_csv_contains_baseline_and_follow_up_rows(self):
        csv_text = session_csv(example_session())
        rows = list(csv.DictReader(StringIO(csv_text)))

        self.assertEqual(
            [
                "record_type",
                "record_id",
                "created_at",
                "external_length_cm",
                "signed_change_cm",
                "absolute_change_cm",
                "indicator_code",
                "confidence",
                "iou",
                "tolerance_cm",
                "imgsz",
            ],
            list(rows[0]),
        )
        self.assertEqual(2, len(rows))
        self.assertEqual("baseline", rows[0]["record_type"])
        self.assertEqual("", rows[0]["signed_change_cm"])
        self.assertEqual("follow_up", rows[1]["record_type"])
        self.assertEqual("0.22", rows[1]["signed_change_cm"])
        self.assertEqual("OUTWARD_INCREASE", rows[1]["indicator_code"])


if __name__ == "__main__":
    unittest.main()
