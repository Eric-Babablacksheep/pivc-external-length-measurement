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
            "camera-input",
            "album-input",
            "image-preview",
            "use-camera-button",
            "choose-album-button",
            "replace-camera-button",
            "replace-album-button",
            "analyse-button",
            "progress-panel",
            "measured-panel",
            "rejected-panel",
            "baseline-diagnostic-image",
        ):
            self.assertRegex(html, rf'id=["\']{re.escape(required_id)}["\']')
        camera_input = re.search(
            r'<input[^>]+id=["\']camera-input["\'][^>]*>', html
        )
        album_input = re.search(
            r'<input[^>]+id=["\']album-input["\'][^>]*>', html
        )
        self.assertIsNotNone(camera_input)
        self.assertIsNotNone(album_input)
        self.assertIn('capture="environment"', camera_input.group(0))
        self.assertNotIn("capture=", album_input.group(0))
        self.assertIn('accept="image/jpeg,image/png"', camera_input.group(0))
        self.assertIn('accept="image/jpeg,image/png"', album_input.group(0))

    def test_camera_and_album_choices_share_image_selection_flow(self):
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        for token in (
            "openImagePicker(elements.cameraInput)",
            "openImagePicker(elements.albumInput)",
            'elements.cameraInput.addEventListener("change"',
            'elements.albumInput.addEventListener("change"',
        ):
            self.assertIn(token, javascript)
        self.assertGreaterEqual(javascript.count("selectImage(event.target.files?.[0])"), 2)

    def test_picker_clears_previous_value_before_reopening(self):
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("function openImagePicker(input)", javascript)
        self.assertIn('input.value = ""', javascript)
        self.assertIn("openImagePicker(elements.cameraInput)", javascript)
        self.assertIn("openImagePicker(elements.albumInput)", javascript)

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
            "continue-follow-up-button",
            "capture-card",
            "comparison-panel",
            "baseline-length-comparison",
            "follow-up-length-comparison",
            "signed-change",
            "absolute-change",
            "baseline-diagnostic-image",
            "follow-up-diagnostic-image",
        ):
            self.assertRegex(html, rf'id=["\']{re.escape(required_id)}["\']')

    def test_results_action_returns_user_to_follow_up_acquisition(self):
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        for token in (
            "function goToFollowUpCapture()",
            "prepareFollowUpCapture()",
            'scrollIntoView({ behavior: "smooth", block: "start" })',
            "elements.cameraButton.focus()",
            'elements.continueFollowUpButton.addEventListener("click"',
            'elements.continueFollowUpButton.textContent = "Add follow-up image"',
            'elements.continueFollowUpButton.textContent = "Add another follow-up"',
        ):
            self.assertIn(token, javascript)

    def test_rejected_baseline_exposes_a_clean_retry_route(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        self.assertRegex(html, r'id=["\']retry-baseline-button["\']')
        for token in (
            "function retryRejectedBaseline()",
            "resetForNewSession()",
            'scrollIntoView({ behavior: "smooth", block: "start" })',
            "elements.cameraButton.focus()",
            "setHidden(elements.retryBaselineButton, followUp)",
            'elements.retryBaselineButton.addEventListener("click"',
        ):
            self.assertIn(token, javascript)

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

    def test_dashboard_has_four_accessible_views(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for name in ("analyse", "sessions", "settings", "guide"):
            self.assertRegex(
                html,
                rf'id=["\']{name}-tab["\'][^>]+role=["\']tab["\']',
            )
            self.assertRegex(
                html,
                rf'id=["\']{name}-view["\'][^>]+role=["\']tabpanel["\']',
            )
        self.assertRegex(html, r'id=["\']mobile-navigation["\']')

    def test_javascript_supports_keyboard_and_hash_navigation(self):
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        for token in (
            "function activateView",
            "aria-selected",
            "location.hash",
            "hashchange",
            "ArrowLeft",
            "ArrowRight",
        ):
            self.assertIn(token, javascript)

    def test_settings_view_exposes_editable_and_runtime_controls(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for required_id in (
            "settings-form",
            "setting-confidence",
            "setting-iou",
            "setting-tolerance",
            "settings-save-button",
            "settings-reset-button",
            "settings-apply-note",
            "runtime-imgsz",
            "runtime-device",
            "runtime-model-status",
        ):
            self.assertRegex(html, rf'id=["\']{required_id}["\']')

    def test_sessions_view_exposes_history_states_and_actions(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        for required_id in (
            "session-list",
            "sessions-loading",
            "sessions-empty",
            "sessions-error",
            "sessions-refresh-button",
        ):
            self.assertRegex(html, rf'id=["\']{required_id}["\']')
        self.assertIn("showBaselineSession(session, false)", javascript)

        javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        for token in (
            "async function loadSettings",
            "async function saveSettings",
            "async function resetSettings",
            "async function loadSessions",
            "async function openSession",
            'localStorage.setItem("pivc-settings"',
            "/export.json",
            "/export.csv",
        ):
            self.assertIn(token, javascript)

    def test_laptop_presentation_has_kpis_and_technical_details(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for required_id in (
            "baseline-length-comparison",
            "follow-up-length-comparison",
            "signed-change",
            "absolute-change",
            "technical-details",
            "technical-settings",
            "technical-session-id",
            "technical-timestamp",
        ):
            self.assertRegex(html, rf'id=["\']{required_id}["\']')
        self.assertRegex(
            html,
            r'<details[^>]+id=["\']technical-details["\']',
        )

    def test_guide_covers_method_and_limitations(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for required_id in (
            "guide-acquisition",
            "guide-method",
            "guide-indicators",
            "guide-rejection",
            "guide-limitations",
            "guide-non-clinical",
        ):
            self.assertRegex(html, rf'id=["\']{required_id}["\']')


if __name__ == "__main__":
    unittest.main()
