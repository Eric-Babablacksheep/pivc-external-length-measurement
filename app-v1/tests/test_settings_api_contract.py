from __future__ import annotations

import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]


class SettingsApiContractTests(unittest.TestCase):
    def test_settings_routes_and_models_are_declared(self):
        source = (APP_DIR / "main.py").read_text(encoding="utf-8")

        self.assertIn('@app.get("/api/v1/settings")', source)
        self.assertIn('@app.put("/api/v1/settings")', source)
        self.assertIn('@app.post("/api/v1/settings/reset")', source)
        self.assertIn("class SettingsUpdate(BaseModel)", source)
        self.assertIn("def validation_config_for", source)

    def test_baseline_and_follow_up_use_session_settings(self):
        source = (APP_DIR / "main.py").read_text(encoding="utf-8")

        self.assertIn("settings_snapshot = SETTINGS_STORE.get()", source)
        self.assertIn('ResearchSettings(**session["settings"])', source)
        self.assertIn("settings=settings_snapshot", source)


if __name__ == "__main__":
    unittest.main()
