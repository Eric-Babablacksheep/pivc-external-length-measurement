# PIVC Tabbed Presentation Dashboard Design

## Objective

Transform the existing single-page PIVC research demonstrator into one responsive PWA that works as a laptop presentation dashboard and a phone image-acquisition interface. The application will expose analysis, temporary session history, research settings, exports, and explanatory guidance without changing the trained segmentation model or claiming clinical validation.

## Scope

The dashboard will provide four views:

1. **Analyse** for baseline and follow-up acquisition, measurement, comparison, diagnostic overlays, and the research change indicator.
2. **Sessions** for reopening measurements created during the current server run and exporting their data.
3. **Settings** for adjusting validated research parameters and viewing read-only model/runtime information.
4. **Guide** for acquisition instructions, method explanation, interpretation, and limitations.

The application remains a local research demonstration. It will not add accounts, patient identifiers, cloud synchronization, a permanent database, clinical alerts, or a live browser-controlled camera.

## Navigation and responsive layout

On laptop screens, a compact header will contain horizontal navigation tabs, application status, and the research-prototype label. The Analyse view will use a persistent acquisition column and a wider results column. After a follow-up succeeds, the results column will prioritize summary cards, the research indicator, and equally sized baseline and follow-up diagnostic overlays.

On phone screens, navigation will become a fixed or sticky bottom navigation bar. Each view will use a single vertical column. The Analyse view will prioritize capture, confirmation, the external-length result, and the research indicator; technical details and diagnostic evidence will remain expandable.

Navigation will use ordinary browser-side state and accessible buttons or links. It will not introduce a frontend framework. The currently selected view may be represented in the URL hash so refresh and browser navigation remain predictable.

## Analyse view

The existing Stages 1–6 workflow will be preserved:

- acquisition guidance and rear-camera-capable file input;
- preview confirmation gate;
- baseline-session creation;
- one or more follow-up measurements;
- signed and absolute length change;
- research change indicator;
- rejected-image handling that preserves the baseline;
- start-new-session action.

The laptop presentation layout will show four summary cards: baseline length, follow-up length, signed change, and absolute change. The research indicator will appear immediately below them. Baseline and follow-up diagnostic overlays will be the principal visual evidence. Model confidence, mark count, centreline length, calibration, mark spacing, session ID, timestamps, and the settings snapshot will be placed in an expandable technical-details area.

## Settings model

The adjustable research defaults will be:

- PIVC confidence threshold, default `0.50`, accepted range `0.05` through `0.95`.
- IoU threshold, default `0.70`, accepted range `0.10` through `0.95`.
- Research change tolerance, default `0.10 cm`, accepted range `0.01` through `1.00 cm`.

Image size will remain fixed at `960` and be displayed read-only because earlier experiments showed that changing inference size alters prediction behavior. Device, model filename, model task, model classes, application version, and model-loaded status will also be read-only.

Settings entered in the browser will be validated by the backend and stored as the server's defaults for new sessions. The browser may mirror the latest accepted defaults in `localStorage` for convenience, but the backend remains authoritative.

### Session consistency rule

When a baseline is established, the backend will store an immutable settings snapshot with that session. Every follow-up in the session will use the same confidence, IoU, image size, and research tolerance. Changing dashboard defaults during an active session affects only a new baseline session. This prevents parameter changes from being mistaken for physical PIVC-length changes.

A reset action will restore the validated defaults. The settings view and each result will clearly distinguish research parameters from clinical thresholds.

## Backend settings interface

The backend will expose:

- `GET /api/v1/settings` to return editable defaults and read-only runtime information.
- `PUT /api/v1/settings` to validate and replace editable defaults.
- `POST /api/v1/settings/reset` to restore validated defaults.

Settings will be represented by a small thread-safe in-memory store. Each measurement call will receive an immutable configuration snapshot rather than reading a mutable global configuration during inference.

The existing direct measurement endpoint will use the current default snapshot. Baseline creation will measure with and store the current snapshot. Follow-up measurement will retrieve and use the session snapshot.

## Sessions view and storage

Sessions remain in memory and are cleared when FastAPI restarts. Each stored session will include:

- session ID and creation time;
- baseline measurement;
- immutable settings snapshot;
- ordered successful follow-up records;
- signed and absolute changes;
- research indicator for each follow-up;
- diagnostic URLs where available.

Rejected baselines will not create sessions. Rejected follow-ups will not enter successful history.

The backend will expose:

- `GET /api/v1/sessions` for a compact newest-first session summary list.
- The existing `GET /api/v1/sessions/{session_id}` for complete details.
- `GET /api/v1/sessions/{session_id}/export.json` for a complete JSON download.
- `GET /api/v1/sessions/{session_id}/export.csv` for a flat baseline/follow-up measurement record.

The Sessions view will show session time, baseline length, successful follow-up count, latest change, latest indicator, and actions to reopen or export a session. Reopening is read-only unless the user explicitly chooses to continue that session with another follow-up.

## Guide view

The Guide view will explain:

- recommended image acquisition;
- PIVC and mark segmentation;
- continuous centreline reconstruction;
- two-consecutive-mark pixel-to-centimetre calibration;
- signed versus absolute change;
- research indicator meanings;
- rejection behavior;
- known limitations and non-clinical status.

The explanation will use concise text and the application's existing visual language. No medical recommendation or patient-care instruction will be presented.

## Error handling and safeguards

- Invalid settings return HTTP `422` with field-specific validation details.
- An unknown session returns HTTP `404`.
- Exporting an unknown session returns HTTP `404`.
- A rejected image displays the existing rejection stage and reason.
- The baseline remains intact after a rejected follow-up.
- The UI warns before starting a new session when an active baseline exists.
- The UI shows when edited defaults apply only to the next session.
- Diagnostic download controls are hidden when no diagnostic file exists.
- All research-indicator displays retain the non-clinical warning.

## Accessibility and presentation behavior

- Tabs use correct selected-state semantics and keyboard navigation.
- Status changes use polite live regions.
- Controls have visible focus styles and descriptive labels.
- Color is supplemented by text and symbols.
- Dashboard cards avoid horizontal scrolling at supported widths.
- Laptop layout targets common `1366×768` and larger presentation screens.
- Phone layout remains usable from approximately `360 px` wide.

## Testing strategy

Backend tests will cover:

- settings validation boundaries and reset;
- immutable settings snapshots;
- follow-ups using their baseline settings after defaults change;
- session summary ordering;
- JSON and CSV export contents;
- unknown-session behavior;
- preservation of the baseline after rejection.

Frontend tests will cover:

- four accessible navigation views;
- laptop and phone navigation structures;
- Analyse state transitions;
- settings validation and reset interactions;
- next-session-only settings messaging;
- session list rendering and reopening;
- export links;
- technical-details disclosure;
- absence of missing selectors and duplicate IDs;
- JavaScript syntax and service-worker cache version.

Existing measurement, session-store, and interface regression tests must remain green.

## Files and components

Expected backend changes:

- `app/main.py`
- `app/session_store.py`
- a focused settings-store/configuration module under `app/`
- backend unit tests under `app/tests/`

Expected frontend changes:

- `app/static/index.html`
- `app/static/styles.css`
- `app/static/app.js`
- `app/static/service-worker.js`
- frontend tests under `app/tests/`

The trained model, `pivc_validation.py`, `pivc_centerline.py`, and the validated post-processing notebook remain unchanged.

## Stage boundary

Completion of Stage 7 means the local PWA provides a responsive tabbed dashboard with analysis, temporary sessions, validated research settings, exports, and guidance. Deployment, authentication, permanent storage, clinical validation, and regulatory readiness remain outside this stage.
