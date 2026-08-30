from __future__ import annotations

import re
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_DIR / "static"


class WebInterfaceStructureTests(unittest.TestCase):
    def test_stage_two_files_exist(self):
        expected = {
            "index.html",
            "styles.css",
            "app.js",
            "manifest.json",
            "service-worker.js",
        }
        self.assertEqual(
            expected,
            {path.name for path in STATIC_DIR.iterdir() if path.is_file()},
        )

    def test_html_exposes_capture_preview_and_result_states(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for required_id in (
            "image-input",
            "image-preview",
            "choose-image-button",
            "replace-image-button",
            "analyse-button",
            "progress-panel",
            "measured-panel",
            "rejected-panel",
            "baseline-diagnostic-image",
        ):
            self.assertRegex(html, rf'id=["\']{re.escape(required_id)}["\']')
        self.assertIn('capture="environment"', html)
        self.assertIn('accept="image/jpeg,image/png"', html)

    def test_javascript_establishes_baseline_and_submits_follow_up(self):
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn('"/api/v1/sessions/baseline"', javascript)
        self.assertIn("/follow-up", javascript)
        self.assertIn('formData.append("image"', javascript)
        self.assertIn('session_status === "BASELINE_ESTABLISHED"', javascript)
        self.assertIn('session_status === "FOLLOW_UP_MEASURED"', javascript)
        self.assertIn('session_status === "FOLLOW_UP_REJECTED"', javascript)
        self.assertIn("diagnostic_url", javascript)

    def test_html_exposes_follow_up_comparison_and_new_session_controls(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for required_id in (
            "capture-heading",
            "result-heading",
            "start-new-session-button",
            "comparison-panel",
            "baseline-length-comparison",
            "follow-up-length-comparison",
            "signed-change",
            "absolute-change",
            "baseline-diagnostic-image",
            "follow-up-diagnostic-image",
        ):
            self.assertRegex(html, rf'id=["\']{re.escape(required_id)}["\']')

    def test_main_registers_follow_up_endpoint(self):
        source = (APP_DIR / "main.py").read_text(encoding="utf-8")
        self.assertIn('/api/v1/sessions/{session_id}/follow-up', source)

    def test_main_serves_static_interface_without_removing_docs(self):
        source = (APP_DIR / "main.py").read_text(encoding="utf-8")
        self.assertRegex(source, r'app\.mount\(\s*["\']/static["\']')
        self.assertIn('FileResponse(STATIC_DIR / "index.html")', source)
        self.assertIn('docs_url="/docs"', source)

    def test_capture_flow_exposes_acquisition_and_confirmation_guidance(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for required_id in (
            "acquisition-guidance",
            "follow-up-guidance",
            "acquisition-confirmation",
            "confirm-full-pivc",
            "confirm-marks-visible",
            "confirm-image-clear",
        ):
            self.assertRegex(html, rf'id=["\']{re.escape(required_id)}["\']')

        analyse_button = re.search(
            r'<button[^>]+id=["\']analyse-button["\'][^>]*>',
            html,
        )
        self.assertIsNotNone(analyse_button)
        self.assertIn("disabled", analyse_button.group(0))

    def test_javascript_requires_all_acquisition_confirmations(self):
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("updateAnalysisReadiness", javascript)
        self.assertIn("acquisitionChecks.every", javascript)
        self.assertIn("checkbox.checked", javascript)
        self.assertIn("elements.analyseButton.disabled = !ready", javascript)

    def test_follow_up_mode_displays_matching_guidance(self):
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("elements.followUpGuidance", javascript)
        self.assertIn("setHidden(elements.followUpGuidance, false)", javascript)


if __name__ == "__main__":
    unittest.main()
