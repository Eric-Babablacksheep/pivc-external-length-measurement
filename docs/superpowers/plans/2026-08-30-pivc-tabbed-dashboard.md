# PIVC Tabbed Presentation Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a responsive four-tab PIVC research dashboard with analysis, immutable per-session settings, temporary session history, exports, and method guidance.

**Architecture:** Keep FastAPI and the existing vanilla HTML/CSS/JavaScript PWA. Add a focused thread-safe settings store, snapshot settings into every baseline session, pass immutable configuration into measurement calls, and extend the in-memory session store for summaries and exports. Convert the existing page into accessible views without changing the trained model or post-processing modules.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, Ultralytics, vanilla JavaScript, HTML5, CSS3, `unittest`, Node syntax checking.

**Spec:** `docs/superpowers/specs/2026-08-30-pivc-tabbed-dashboard-design.md`

## Global Constraints

- The application remains a local research demonstration and must retain the non-clinical warning.
- Editable defaults are confidence `0.50`, IoU `0.70`, and tolerance `0.10 cm`.
- Accepted ranges are confidence `0.05–0.95`, IoU `0.10–0.95`, and tolerance `0.01–1.00 cm`.
- Image size remains fixed and read-only at `960`.
- Baseline sessions store an immutable settings snapshot; all follow-ups use that snapshot.
- Settings changes affect new sessions only.
- Sessions remain in memory and are cleared on FastAPI restart.
- Do not modify `best.pt`, `pi-cent_conv/pivc_validation.py`, `pi-cent_conv/pivc_centerline.py`, or the post-processing notebook.
- Rejected follow-ups never alter the baseline or successful history.
- Laptop layout targets `1366×768` and larger; phone layout remains usable at approximately `360 px`.

---

### Task 1: Thread-safe research settings store

**Files:**
- Create: `app/settings_store.py`
- Create: `app/tests/test_settings_store.py`

**Interfaces:**
- Produces: `ResearchSettings(confidence: float, iou: float, tolerance_cm: float, imgsz: int = 960)`.
- Produces: `ResearchSettingsStore.get() -> ResearchSettings`, `update(...) -> ResearchSettings`, and `reset() -> ResearchSettings`.
- Validation raises `ValueError` with the invalid field name.

- [ ] **Step 1: Write failing settings-store tests**

```python
class ResearchSettingsStoreTests(unittest.TestCase):
    def test_defaults_are_validated_values(self):
        settings = ResearchSettingsStore().get()
        self.assertEqual(0.50, settings.confidence)
        self.assertEqual(0.70, settings.iou)
        self.assertEqual(0.10, settings.tolerance_cm)
        self.assertEqual(960, settings.imgsz)

    def test_update_returns_immutable_snapshot(self):
        store = ResearchSettingsStore()
        snapshot = store.update(confidence=0.35, iou=0.60, tolerance_cm=0.15)
        store.reset()
        self.assertEqual(0.35, snapshot.confidence)

    def test_boundaries_are_enforced(self):
        store = ResearchSettingsStore()
        for field, values in {
            "confidence": (0.04, 0.96),
            "iou": (0.09, 0.96),
            "tolerance_cm": (0.0, 1.01),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    kwargs = {"confidence": 0.5, "iou": 0.7, "tolerance_cm": 0.1}
                    kwargs[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        store.update(**kwargs)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest app.tests.test_settings_store -v`

Expected: import failure because `settings_store.py` does not exist.

- [ ] **Step 3: Implement the immutable store**

```python
from dataclasses import asdict, dataclass
from threading import Lock

@dataclass(frozen=True)
class ResearchSettings:
    confidence: float = 0.50
    iou: float = 0.70
    tolerance_cm: float = 0.10
    imgsz: int = 960

    def to_dict(self) -> dict:
        return asdict(self)

class ResearchSettingsStore:
    def __init__(self):
        self._lock = Lock()
        self._settings = ResearchSettings()

    @staticmethod
    def _validate(confidence: float, iou: float, tolerance_cm: float) -> None:
        bounds = {
            "confidence": (confidence, 0.05, 0.95),
            "iou": (iou, 0.10, 0.95),
            "tolerance_cm": (tolerance_cm, 0.01, 1.00),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")

    def get(self) -> ResearchSettings:
        with self._lock:
            return self._settings

    def update(self, *, confidence: float, iou: float, tolerance_cm: float) -> ResearchSettings:
        self._validate(confidence, iou, tolerance_cm)
        replacement = ResearchSettings(confidence, iou, tolerance_cm)
        with self._lock:
            self._settings = replacement
        return replacement

    def reset(self) -> ResearchSettings:
        with self._lock:
            self._settings = ResearchSettings()
            return self._settings
```

- [ ] **Step 4: Run settings tests and verify GREEN**

Run: `python -m unittest app.tests.test_settings_store -v`

Expected: all settings-store tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add app/settings_store.py app/tests/test_settings_store.py
git commit -m "feat: add validated research settings store"
```

---

### Task 2: Freeze settings into baseline sessions and measurement calls

**Files:**
- Modify: `app/session_store.py`
- Modify: `app/main.py`
- Modify: `app/tests/test_session_store.py`
- Create: `app/tests/test_settings_api_contract.py`

**Interfaces:**
- Consumes: `ResearchSettings` and `ResearchSettingsStore` from Task 1.
- Changes: `BaselineSessionStore.create(baseline_measurement: dict, settings: dict) -> dict`.
- Changes: `BaselineSessionStore.add_follow_up(session_id: str, follow_up_measurement: dict, research_indicator: dict) -> dict | None` stores the indicator with the successful record.
- Changes: `measure_pivc(request, image, settings: ResearchSettings | None = None) -> dict`.
- Produces: settings routes `GET /api/v1/settings`, `PUT /api/v1/settings`, and `POST /api/v1/settings/reset`.

- [ ] **Step 1: Add failing session snapshot tests**

```python
def test_session_preserves_settings_snapshot(self):
    settings = {"confidence": 0.4, "iou": 0.6, "tolerance_cm": 0.15, "imgsz": 960}
    session = self.store.create(measured(3.5), settings=settings)
    settings["confidence"] = 0.9
    self.assertEqual(0.4, session["settings"]["confidence"])

def test_follow_up_does_not_replace_session_settings(self):
    settings = {"confidence": 0.4, "iou": 0.6, "tolerance_cm": 0.15, "imgsz": 960}
    session = self.store.create(measured(3.5), settings=settings)
    indicator = {"code": "WITHIN_TOLERANCE", "label": "Within measurement tolerance"}
    updated = self.store.add_follow_up(session["session_id"], measured(3.4), indicator)
    self.assertEqual(settings, updated["settings"])
    self.assertEqual("WITHIN_TOLERANCE", updated["follow_ups"][0]["research_indicator"]["code"])
```

- [ ] **Step 2: Add failing API contract tests**

Read `app/main.py` as a contract and assert the registered source contains the three exact route paths, a `SettingsUpdate` Pydantic model, and use of a per-call configuration builder. Keep model inference outside these lightweight tests.

```python
def test_settings_routes_and_models_are_declared(self):
    source = (APP_DIR / "main.py").read_text(encoding="utf-8")
    self.assertIn('@app.get("/api/v1/settings")', source)
    self.assertIn('@app.put("/api/v1/settings")', source)
    self.assertIn('@app.post("/api/v1/settings/reset")', source)
    self.assertIn("class SettingsUpdate(BaseModel)", source)
    self.assertIn("def validation_config_for", source)
```

- [ ] **Step 3: Run targeted tests and verify RED**

Run: `python -m unittest app.tests.test_session_store app.tests.test_settings_api_contract -v`

Expected: failures for the missing `settings` argument, routes, and helper.

- [ ] **Step 4: Update session creation**

Change `create` to require a settings dictionary and store `deepcopy(settings)` under `session["settings"]`. Change `add_follow_up` to require a research-indicator dictionary and store its deep copy under the successful entry. Update all test setup and follow-up calls to provide validated settings and an indicator fixture.

- [ ] **Step 5: Add settings API and immutable measurement configuration**

Add `SETTINGS_STORE = ResearchSettingsStore()` and:

```python
class SettingsUpdate(BaseModel):
    confidence: float = Field(ge=0.05, le=0.95)
    iou: float = Field(ge=0.10, le=0.95)
    tolerance_cm: float = Field(ge=0.01, le=1.00)

def validation_config_for(settings: ResearchSettings) -> ValidationConfig:
    return ValidationConfig(
        imgsz=settings.imgsz,
        confidence=settings.confidence,
        iou=settings.iou,
        device="cpu",
        known_mark_spacing_cm=1.0,
        accuracy_tolerance_cm=settings.tolerance_cm,
    )
```

The GET response must include `editable`, `runtime`, and `applies_to: "new_sessions"`. PUT catches no generic exception because Pydantic handles ranges; it calls `SETTINGS_STORE.update`. Reset calls `SETTINGS_STORE.reset`.

- [ ] **Step 6: Route measurements through snapshots**

`measure_pivc` selects `settings or SETTINGS_STORE.get()` and passes `validation_config_for(selected_settings)` to `process_validation_case`. Baseline creation captures one settings snapshot before measurement, passes it to measurement, and stores `settings.to_dict()`. Follow-up retrieves `ResearchSettings(**session["settings"])` and passes that snapshot to measurement. Research classification accepts the session tolerance as an argument instead of reading a mutable global. Before storing a successful follow-up, calculate its signed change from the baseline, classify it with the frozen tolerance, and pass the resulting indicator into `add_follow_up` so session summaries and exports can use the stored evidence.

- [ ] **Step 7: Run targeted and full tests**

Run: `python -m unittest discover -s app/tests -v`

Expected: all tests pass, including existing baseline and follow-up tests.

- [ ] **Step 8: Commit Task 2**

```powershell
git add app/main.py app/session_store.py app/tests/test_session_store.py app/tests/test_settings_api_contract.py
git commit -m "feat: freeze research settings per session"
```

---

### Task 3: Session summaries and JSON/CSV exports

**Files:**
- Modify: `app/session_store.py`
- Modify: `app/main.py`
- Modify: `app/tests/test_session_store.py`
- Create: `app/tests/test_session_exports.py`

**Interfaces:**
- Produces: `BaselineSessionStore.list_summaries() -> list[dict]` newest first.
- Produces: `GET /api/v1/sessions`.
- Produces: `GET /api/v1/sessions/{session_id}/export.json` and `.csv`.

- [ ] **Step 1: Add failing newest-first summary tests**

Create two sessions, add a follow-up to the older session, and assert the summary keys are exactly `session_id`, `created_at`, `baseline_length_cm`, `successful_follow_up_count`, `latest_signed_change_cm`, and `latest_indicator`. Inject deterministic `created_at` values into the in-memory fixtures before asserting order.

- [ ] **Step 2: Add failing export-format tests**

Test a pure helper `session_csv(session: dict) -> str` with literal expected headers:

```text
record_type,record_id,created_at,external_length_cm,signed_change_cm,absolute_change_cm,indicator_code,confidence,iou,tolerance_cm,imgsz
```

Assert one baseline row and one row per follow-up. Assert `session_export_payload` returns a deep-copy-compatible dictionary containing baseline, follow-ups, settings, and warning.

- [ ] **Step 3: Run targeted tests and verify RED**

Run: `python -m unittest app.tests.test_session_store app.tests.test_session_exports -v`

Expected: failures for missing summary and export helpers.

- [ ] **Step 4: Implement summaries and export helpers**

Use `csv.DictWriter` with `io.StringIO(newline="")`. Baseline change fields remain empty. The latest indicator comes from the latest follow-up record. Return deep copies from the store.

- [ ] **Step 5: Add routes**

JSON export returns a `JSONResponse` with `Content-Disposition: attachment`. CSV export returns a `StreamingResponse(iter([csv_text]), media_type="text/csv")`. Both use filename `pivc-session-{session_id}.json|csv` and return `404` for unknown sessions.

- [ ] **Step 6: Run full tests and verify GREEN**

Run: `python -m unittest discover -s app/tests -v`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```powershell
git add app/main.py app/session_store.py app/tests/test_session_store.py app/tests/test_session_exports.py
git commit -m "feat: add session summaries and exports"
```

---

### Task 4: Accessible responsive tab shell

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/styles.css`
- Modify: `app/static/app.js`
- Modify: `app/tests/test_web_interface.py`

**Interfaces:**
- Produces view IDs: `analyse-view`, `sessions-view`, `settings-view`, and `guide-view`.
- Produces tab IDs: `analyse-tab`, `sessions-tab`, `settings-tab`, and `guide-tab`.
- Produces: `activateView(viewName: string, updateHash = true)`.

- [ ] **Step 1: Add failing navigation structure tests**

Assert four buttons have `role="tab"`, matching `aria-controls`, and four panels have `role="tabpanel"`. Assert a mobile navigation container exists. Assert `app.js` includes `activateView`, `aria-selected`, hash handling, ArrowLeft, and ArrowRight.

- [ ] **Step 2: Run navigation tests and verify RED**

Run: `python -m unittest app.tests.test_web_interface -v`

Expected: failures for missing tabs and views.

- [ ] **Step 3: Restructure HTML without changing Analyse element IDs**

Wrap the current analysis workflow in `#analyse-view`. Add empty-but-labeled `#sessions-view`, `#settings-view`, and `#guide-view` panels. Add desktop and mobile navigation controls whose `data-view` values are `analyse`, `sessions`, `settings`, and `guide`.

- [ ] **Step 4: Implement navigation behavior**

`activateView` hides non-selected panels, synchronizes every desktop/mobile control sharing the same `data-view`, updates `aria-selected`, optionally sets `location.hash`, and focuses the next tab on arrow-key navigation. Unknown hashes fall back to Analyse.

- [ ] **Step 5: Implement laptop and phone shell CSS**

At widths above `900px`, show horizontal header tabs and the two-column Analyse dashboard. At `900px` and below, hide desktop tabs, show bottom navigation, add safe-area padding, and stack Analyse content. Preserve all existing `360px` rules and prevent horizontal overflow.

- [ ] **Step 6: Run tests and syntax checks**

Run: `python -m unittest app.tests.test_web_interface -v`

Run: `node --check app/static/app.js`

Expected: tests and syntax check pass.

- [ ] **Step 7: Commit Task 4**

```powershell
git add app/static/index.html app/static/styles.css app/static/app.js app/tests/test_web_interface.py
git commit -m "feat: add responsive dashboard navigation"
```

---

### Task 5: Settings and Sessions views

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/styles.css`
- Modify: `app/static/app.js`
- Modify: `app/tests/test_web_interface.py`

**Interfaces:**
- Consumes settings and session APIs from Tasks 2–3.
- Produces: `loadSettings()`, `saveSettings(event)`, `resetSettings()`, `loadSessions()`, and `openSession(sessionId)`.

- [ ] **Step 1: Add failing view-content tests**

Assert Settings contains inputs `setting-confidence`, `setting-iou`, and `setting-tolerance`, read-only fields for image size/device/model status, save/reset buttons, and `settings-apply-note`. Assert Sessions contains `session-list`, empty/loading/error states, refresh action, and JSON/CSV export anchors.

- [ ] **Step 2: Add failing JavaScript contract tests**

Assert the frontend calls `/api/v1/settings`, `/api/v1/settings/reset`, and `/api/v1/sessions`; renders field errors; stores only accepted settings in `localStorage`; and builds exports from `/api/v1/sessions/${sessionId}/export.json|csv`.

- [ ] **Step 3: Run tests and verify RED**

Run: `python -m unittest app.tests.test_web_interface -v`

Expected: failures for missing controls and functions.

- [ ] **Step 4: Implement Settings view**

Use numeric inputs with backend-matching min/max/step. Disable Save during requests. Render accepted values returned by the server. Display “Applies to new baseline sessions” whenever an active session exists. Reset requires one confirmation and then reloads accepted defaults. Read-only runtime values come from the GET response.

- [ ] **Step 5: Implement Sessions view**

Load newest-first summaries when the tab opens and after a successful baseline/follow-up. Each card shows the six summary fields and buttons for Open, JSON, and CSV. `openSession` fetches details, activates Analyse, renders the baseline and latest follow-up read-only, and offers “Continue this session” to set `activeSessionId` explicitly.

- [ ] **Step 6: Style forms, cards, loading, empty, and error states**

Use existing colors and card primitives. Ensure tap targets are at least `44px`, labels remain visible, and session actions wrap at phone width.

- [ ] **Step 7: Run tests and selector audit**

Run: `python -m unittest discover -s app/tests -v`

Run a Python selector audit that extracts HTML IDs and every `$("#id")` selector, failing on missing selectors or duplicate IDs.

- [ ] **Step 8: Commit Task 5**

```powershell
git add app/static/index.html app/static/styles.css app/static/app.js app/tests/test_web_interface.py
git commit -m "feat: add settings and session dashboard views"
```

---

### Task 6: Laptop result presentation and Guide view

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/styles.css`
- Modify: `app/static/app.js`
- Modify: `app/static/service-worker.js`
- Modify: `app/tests/test_web_interface.py`

**Interfaces:**
- Produces laptop KPI cards and `details#technical-details`.
- Produces complete static Guide content.
- Preserves existing measurement, rejection, diagnostic, and research-indicator element IDs.

- [ ] **Step 1: Add failing presentation tests**

Assert KPI containers exist for baseline, follow-up, signed change, and absolute change. Assert `technical-details` is a semantic `<details>` element containing confidence, marks, centreline, calibration, mark spacing, settings snapshot, session ID, and timestamp. Assert Guide headings cover acquisition, method, indicator interpretation, rejection, limitations, and non-clinical status.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest app.tests.test_web_interface -v`

Expected: failures for the missing dashboard and Guide structures.

- [ ] **Step 3: Implement laptop presentation markup and rendering**

Move existing comparison values into the KPI row while keeping their IDs. Add baseline-only placeholders until a follow-up exists. Keep diagnostic figures equal in layout and hide absent images. Populate the technical settings snapshot from baseline/session responses.

- [ ] **Step 4: Implement Guide content**

Use concise prose consistent with the approved spec. State that the `±0.10 cm` default is measurement tolerance, not a clinically validated dislodgement threshold, and that image acquisition consistency affects reliability.

- [ ] **Step 5: Final responsive styling and service-worker update**

Use CSS grid for the laptop KPI and evidence areas and one-column fallbacks below `900px`. Change cache name to `pivc-vision-stage-7-v1`.

- [ ] **Step 6: Run complete verification**

Run: `python -m unittest discover -s app/tests -v`

Run: `python -m py_compile app/main.py app/session_store.py app/settings_store.py`

Run: `node --check app/static/app.js`

Run the selector/duplicate-ID audit from Task 5.

Expected: all tests, compilation, syntax, and selector checks pass.

- [ ] **Step 7: Manual acceptance checklist**

Using the user's configured Python environment, restart FastAPI and verify at laptop and narrow phone widths:

1. Establish a baseline with validated defaults.
2. Change defaults and confirm the active session still reports its original snapshot.
3. Measure a follow-up and verify KPIs, indicator, and both diagnostics.
4. Reject a follow-up and verify the baseline remains.
5. Open Sessions, reopen the session, and download valid JSON and CSV files.
6. Reset Settings and confirm `0.50`, `0.70`, `0.10`, and `960` are displayed.
7. Navigate all four views using mouse, keyboard arrows, and phone bottom navigation.

- [ ] **Step 8: Commit Task 6**

```powershell
git add app/static/index.html app/static/styles.css app/static/app.js app/static/service-worker.js app/tests/test_web_interface.py
git commit -m "feat: complete PIVC presentation dashboard"
```

---

### Task 7: Final regression review and handoff

**Files:**
- Review: all Stage 7 files
- Modify only if a failing verification demonstrates a defect.

**Interfaces:**
- Consumes all prior tasks.
- Produces a verified Stage 7 dashboard handoff.

- [ ] **Step 1: Review the implementation against every design section**

Create a checklist covering four views, responsive navigation, frozen settings, runtime information, session summaries, exports, Guide content, errors, accessibility, and stage boundary. Record any missing item before changing code.

- [ ] **Step 2: For each discovered defect, add a failing regression test**

Run the smallest relevant test and confirm it fails for the observed defect before applying a fix.

- [ ] **Step 3: Apply only demonstrated fixes and rerun targeted tests**

Do not refactor the model or post-processing pipeline.

- [ ] **Step 4: Run fresh full verification**

```powershell
python -m unittest discover -s app/tests -v
python -m py_compile app/main.py app/session_store.py app/settings_store.py
node --check app/static/app.js
```

Expected: zero failures and zero syntax errors.

- [ ] **Step 5: Commit verified corrections if any**

```powershell
git add app docs/superpowers/plans/2026-08-30-pivc-tabbed-dashboard.md
git commit -m "test: verify PIVC tabbed dashboard"
```
