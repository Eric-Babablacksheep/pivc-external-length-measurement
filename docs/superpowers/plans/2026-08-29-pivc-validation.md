# PIVC Measurement Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable 40-image PIVC measurement validation pipeline that produces diagnostic overlays and a verified Excel workbook reporting algorithm reliability at a ±0.10 cm accuracy threshold.

**Architecture:** A focused Python module discovers cases, performs one-image inference and post-processing, classifies outcomes, saves per-image diagnostics, and checkpoints JSON after every case. A thin Jupyter notebook invokes that module. A separate JavaScript builder consumes the checkpoint and uses `@oai/artifact-tool` to create and visually verify the six-sheet Excel workbook.

**Tech Stack:** Python 3.11, Ultralytics YOLOv8 segmentation, OpenCV, NumPy, SciPy, scikit-image, NetworkX, Matplotlib, `unittest`, Jupyter, JavaScript, `@oai/artifact-tool` 2.8.6+.

**Spec:** `docs/superpowers/specs/2026-08-29-pivc-validation-design.md`

## Global Constraints

- Dataset root: `pi-cent_conv/test_images`.
- Expected structure: L35/L40/L45/L50, each containing `with_dressing` and `without_dressing`, each containing exactly five images.
- Known lengths: L35=3.5 cm, L40=4.0 cm, L45=4.5 cm, L50=5.0 cm.
- Fixed inference configuration: CPU, `imgsz=960`, `conf=0.50`, `iou=0.70`, `retina_masks=True`.
- Every image uses the same configuration; no per-image threshold changes.
- Consecutive graduation-mark centroids represent exactly 1.0 cm.
- `ACCURATE` means absolute error ≤0.10 cm; `NOT ACCURATE` means absolute error >0.10 cm; unsafe or incomplete measurements are `REJECTED`.
- Rejected images remain in total-image denominators.
- The workbook `Image Results` sheet contains exactly the twelve approved columns.
- `Endpoints on PIVC line` is manual and defaults to `Unreviewed`, or `N/A` when no centreline exists.
- Workbook authoring must use the loader-provided `@oai/artifact-tool`; do not use `openpyxl`, `xlsxwriter`, or `pandas.ExcelWriter`.
- The final workbook is descriptive algorithm validation on phantom images, not evidence of clinical validity.

---

## File Structure

- Create `pi-cent_conv/pivc_validation.py`: validation dataclasses, folder discovery, segmentation adaptation, component extraction, one-image processing, status classification, diagnostic rendering, checkpointing, and batch orchestration.
- Create `pi-cent_conv/test_pivc_validation.py`: unit and integration tests for every validation boundary.
- Create `pi-cent_conv/run_pivc_validation.ipynb`: thin local CPU notebook for configuration, run/resume, progress display, and workbook invocation.
- Create `pi-cent_conv/build_validation_workbook.mjs`: artifact-tool workbook generation and compact inspection/render/export verification.
- Modify `pi-cent_conv/pivc_centerline.py` only if a tested reusable primitive is missing; do not duplicate existing reconstruction, repair, or calibration logic.
- Generate `pi-cent_conv/validation_outputs/checkpoint.json`: resumable machine-readable results.
- Generate `pi-cent_conv/validation_outputs/diagnostics/<case_id>.jpg`: one diagnostic overlay per attempted case when possible.
- Generate `outputs/pivc_validation_<run_id>/PIVC_Validation.xlsx`: final workbook.

---

### Task 1: Validation Case Discovery and Dataset Gate

**Files:**
- Create: `pi-cent_conv/pivc_validation.py`
- Create: `pi-cent_conv/test_pivc_validation.py`

**Interfaces:**
- Produces: `ValidationConfig`, `ValidationCase`, `discover_validation_cases(root: Path) -> list[ValidationCase]`, `validate_case_balance(cases: Sequence[ValidationCase]) -> None`.
- Consumes: folder layout and fixed constants from the approved specification.

- [ ] **Step 1: Write failing metadata tests**

```python
class DatasetDiscoveryTests(unittest.TestCase):
    def test_discovers_length_and_dressing_from_folders(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "L35" / "with_dressing" / "sample.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"not-decoded-in-this-test")
            cases = discover_validation_cases(root)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].known_length_cm, 3.5)
            self.assertEqual(cases[0].dressing_condition, "with_dressing")

    def test_balance_gate_requires_five_images_per_subgroup(self):
        with self.assertRaisesRegex(RuntimeError, "expected 5"):
            validate_case_balance([
                ValidationCase(Path("L35/with_dressing/one.jpg"), "L35", 3.5, "with_dressing")
            ])
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run: `python -m unittest -v test_pivc_validation.DatasetDiscoveryTests`

Expected: FAIL because `pivc_validation` and its interfaces do not exist.

- [ ] **Step 3: Implement immutable configuration and discovery**

```python
@dataclass(frozen=True)
class ValidationConfig:
    imgsz: int = 960
    confidence: float = 0.50
    iou: float = 0.70
    device: str = "cpu"
    known_mark_spacing_cm: float = 1.0
    accuracy_tolerance_cm: float = 0.10

@dataclass(frozen=True)
class ValidationCase:
    image_path: Path
    length_group: str
    known_length_cm: float
    dressing_condition: str

LENGTHS_CM = {"L35": 3.5, "L40": 4.0, "L45": 4.5, "L50": 5.0}
DRESSING_GROUPS = ("with_dressing", "without_dressing")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
```

Discovery must sort by length group, dressing group, and filename; reject unknown folders, duplicate resolved paths, and non-balanced subgroup counts.

- [ ] **Step 4: Run discovery tests and the real dataset gate**

Run: `python -m unittest -v test_pivc_validation.DatasetDiscoveryTests`

Expected: PASS.

Run: `python -c "from pathlib import Path; from pivc_validation import discover_validation_cases,validate_case_balance; c=discover_validation_cases(Path('test_images')); validate_case_balance(c); print(len(c))"`

Expected: `40`.

- [ ] **Step 5: Commit**

```bash
git add pi-cent_conv/pivc_validation.py pi-cent_conv/test_pivc_validation.py
git commit -m "feat: validate PIVC test dataset structure"
```

---

### Task 2: Structured Results, Accuracy Classification, and Rejections

**Files:**
- Modify: `pi-cent_conv/pivc_validation.py`
- Modify: `pi-cent_conv/test_pivc_validation.py`

**Interfaces:**
- Consumes: `ValidationCase`, `ValidationConfig` from Task 1.
- Produces: `ValidationResult`, `classify_accuracy(estimated_cm: float, known_cm: float, tolerance_cm: float) -> tuple[str, float]`, `rejected_result(case, stage, reason) -> ValidationResult`.

- [ ] **Step 1: Write failing classification tests**

```python
class AccuracyClassificationTests(unittest.TestCase):
    def test_exact_tolerance_is_accurate(self):
        status, error = classify_accuracy(3.6, 3.5, 0.10)
        self.assertEqual(status, "ACCURATE")
        self.assertAlmostEqual(error, 0.10)

    def test_error_above_tolerance_is_not_accurate(self):
        status, error = classify_accuracy(3.601, 3.5, 0.10)
        self.assertEqual(status, "NOT ACCURATE")
        self.assertGreater(error, 0.10)

    def test_rejection_preserves_case_identity_and_reason(self):
        result = rejected_result(case, "calibration", "only one mark")
        self.assertEqual(result.status, "REJECTED")
        self.assertEqual(result.rejection_stage, "calibration")
        self.assertIn("only one mark", result.rejection_reason)
```

- [ ] **Step 2: Verify the tests fail for missing result interfaces**

Run: `python -m unittest -v test_pivc_validation.AccuracyClassificationTests`

Expected: FAIL because the result interfaces are not defined.

- [ ] **Step 3: Implement a JSON-safe result record**

```python
@dataclass
class ValidationResult:
    case_id: str
    filename: str
    image_path: str
    length_group: str
    dressing_condition: str
    known_length_cm: float
    pivc_detected: bool = False
    pivc_confidence: float | None = None
    marks_detected: int = 0
    endpoints_review: str = "N/A"
    corrected_centreline_px: float | None = None
    consecutive_mark_spacings_px: list[float] = field(default_factory=list)
    pixels_per_cm: float | None = None
    estimated_length_cm: float | None = None
    absolute_error_cm: float | None = None
    status: str = "REJECTED"
    rejection_stage: str = ""
    rejection_reason: str = ""
    diagnostic_path: str = ""
    diagnostic_available: bool = False
    config_fingerprint: str = ""
```

Use `Decimal(str(value))` or a `1e-9` comparison guard so binary floating-point does not misclassify an error mathematically equal to 0.10 cm.

- [ ] **Step 4: Run all result tests**

Run: `python -m unittest -v test_pivc_validation.AccuracyClassificationTests`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pi-cent_conv/pivc_validation.py pi-cent_conv/test_pivc_validation.py
git commit -m "feat: add validation result classification"
```

---

### Task 3: Single-Image Validation Pipeline

**Files:**
- Modify: `pi-cent_conv/pivc_validation.py`
- Modify: `pi-cent_conv/test_pivc_validation.py`
- Reuse: `pi-cent_conv/pivc_centerline.py`

**Interfaces:**
- Consumes: `reconstruct_continuous_centerline(component_paths)`, `repair_centerline_through_marks(path, mark_centres, pivc_width)`, `calibrate_length_from_marks(path, mark_centres, pivc_width, known_spacing_cm=1.0)`.
- Produces: `SegmentationOutput`, `predict_segmentation(model, image_path, config)`, `component_paths_from_mask(mask)`, `process_validation_case(case, model, config, diagnostics_dir) -> ValidationResult`.

- [ ] **Step 1: Write failing synthetic-mask and failure-stage tests**

```python
class SingleImagePipelineTests(unittest.TestCase):
    def test_component_extraction_returns_paths_and_widths(self):
        mask = np.zeros((240, 120), dtype=np.uint8)
        cv2.rectangle(mask, (55, 10), (65, 100), 1, -1)
        cv2.rectangle(mask, (55, 110), (65, 220), 1, -1)
        components = component_paths_from_mask(mask)
        self.assertEqual(len(components), 2)
        self.assertTrue(all(item["median_width"] > 0 for item in components))

    def test_process_rejects_fewer_than_two_marks_without_raising(self):
        segmentation = synthetic_segmentation(mark_centres=[(100, 60)])
        result = process_validation_case(
            case, model=None, config=ValidationConfig(),
            diagnostics_dir=Path(self.directory), predictor=lambda *_: segmentation,
        )
        self.assertEqual(result.status, "REJECTED")
        self.assertEqual(result.rejection_stage, "calibration")
```

- [ ] **Step 2: Run the single-image tests and verify failure**

Run: `python -m unittest -v test_pivc_validation.SingleImagePipelineTests`

Expected: FAIL because segmentation adaptation and processing are missing.

- [ ] **Step 3: Extract notebook post-processing into focused functions**

Implement:

```python
@dataclass
class SegmentationOutput:
    image_rgb: np.ndarray
    pivc_confidence: float
    pivc_mask: np.ndarray
    mark_masks: list[np.ndarray]
```

Implement `component_paths_from_mask(mask: np.ndarray) -> list[dict]` by applying `cv2.connectedComponentsWithStats(mask, connectivity=8)`, retaining components whose area is at least `max(50, int(0.01 * largest_area))`, converting each component skeleton to a weighted 8-neighbour graph, selecting its longest endpoint-to-endpoint path, and calculating median width from `2 * distance_transform_edt(component_mask)` sampled along that path. Implement `process_segmentation(case: ValidationCase, output: SegmentationOutput, config: ValidationConfig) -> tuple[ValidationResult, dict]` by passing those component records to `reconstruct_continuous_centerline`, then `repair_centerline_through_marks`, then `calibrate_length_from_marks`, and finally `classify_accuracy`.

Reuse the exact rotation-independent chaining, mark repair, and calibration behavior already tested in `pivc_centerline.py`. Do not copy an earlier orientation-dependent implementation.

- [ ] **Step 4: Implement the Ultralytics adapter with fixed settings**

```python
def predict_segmentation(model, image_path, config):
    result = model.predict(
        source=str(image_path), imgsz=config.imgsz,
        conf=config.confidence, iou=config.iou,
        device=config.device, retina_masks=True,
        max_det=100, verbose=False,
    )[0]
```

Require model task `segment` and class names `mark` and `picc`. Return a structured rejection for missing masks, no usable PIVC, multiple non-overlapping PIVCs, or unreadable images. Do not let one exception terminate the batch.

- [ ] **Step 5: Run all single-image and existing centreline tests**

Run: `python -m unittest -v test_pivc_validation.SingleImagePipelineTests test_pivc_centerline.py`

Expected: PASS.

- [ ] **Step 6: Run one real-image integration check**

Run: `python -m unittest -v test_pivc_validation.RealImageIntegrationTests`

The test loads `best.pt` and one existing validation image, asserts that processing returns either a complete measured result or a structured `REJECTED` result, and verifies that no unhandled exception escapes.

- [ ] **Step 7: Commit**

```bash
git add pi-cent_conv/pivc_validation.py pi-cent_conv/test_pivc_validation.py
git commit -m "feat: process one PIVC validation image"
```

---

### Task 4: Diagnostic Overlay With Mark-Distance Annotation

**Files:**
- Modify: `pi-cent_conv/pivc_validation.py`
- Modify: `pi-cent_conv/test_pivc_validation.py`

**Interfaces:**
- Consumes: processing diagnostics containing masks, component paths, corrected path, projected marks, spacing values, endpoints, and status.
- Produces: `render_validation_overlay(case, result, diagnostics, output_path) -> Path`.

- [ ] **Step 1: Write failing overlay tests**

```python
class DiagnosticOverlayTests(unittest.TestCase):
    def test_success_overlay_is_saved_at_readable_resolution(self):
        output = render_validation_overlay(case, result, diagnostics, output_path)
        image = cv2.imread(str(output))
        self.assertIsNotNone(image)
        self.assertGreaterEqual(image.shape[1], 1200)
        self.assertTrue(result.diagnostic_available)

    def test_rejected_overlay_includes_last_available_stage(self):
        output = render_validation_overlay(case, rejected, partial_diagnostics, output_path)
        self.assertTrue(output.is_file())
```

- [ ] **Step 2: Verify overlay tests fail**

Run: `python -m unittest -v test_pivc_validation.DiagnosticOverlayTests`

Expected: FAIL because the renderer is absent.

- [ ] **Step 3: Implement a single audit-friendly overlay**

Use OpenCV RGB drawing and save as quality-controlled JPEG or PNG. Draw:

- translucent PIVC mask;
- corrected centreline;
- reconstructed bridges and mark-repaired sections;
- original and projected mark centres;
- each consecutive projected-mark centreline interval with `N.NN px = 1.00 cm`;
- terminal endpoints;
- filename, dressing condition, known length, estimated length, absolute error, and status;
- rejection stage and reason for rejected cases.

Resize only for output display; retain coordinates in the original image system. Preserve aspect ratio.

- [ ] **Step 4: Run overlay tests and inspect one saved image**

Run: `python -m unittest -v test_pivc_validation.DiagnosticOverlayTests`

Expected: PASS.

Open the generated test overlay and verify that the distance line and text do not obscure the PIVC.

- [ ] **Step 5: Commit**

```bash
git add pi-cent_conv/pivc_validation.py pi-cent_conv/test_pivc_validation.py
git commit -m "feat: render PIVC validation diagnostics"
```

---

### Task 5: Checkpointing, Resume, and Batch Orchestration

**Files:**
- Modify: `pi-cent_conv/pivc_validation.py`
- Modify: `pi-cent_conv/test_pivc_validation.py`

**Interfaces:**
- Consumes: `process_validation_case`, discovered cases, model, and fixed config.
- Produces: `configuration_fingerprint(config: ValidationConfig, model_path: Path) -> str`, `load_checkpoint(path: Path) -> dict`, `save_checkpoint_atomic(path: Path, payload: dict) -> None`, `run_validation_batch(cases: Sequence[ValidationCase], model, config: ValidationConfig, output_dir: Path, processor=process_validation_case) -> list[ValidationResult]`.

- [ ] **Step 1: Write failing checkpoint tests**

```python
class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_results(self):
        save_checkpoint_atomic(path, {"fingerprint": "abc", "results": [asdict(result)]})
        loaded = load_checkpoint(path)
        self.assertEqual(loaded["results"][0]["filename"], result.filename)

    def test_resume_skips_only_complete_matching_cases(self):
        processed = []
        run_validation_batch(cases, model, config, output_dir,
            processor=lambda case, *_: processed.append(case.case_id) or completed(case))
        self.assertEqual(processed, [second_case.case_id])

    def test_changed_configuration_invalidates_resume(self):
        self.assertNotEqual(
            configuration_fingerprint(ValidationConfig(confidence=0.50), model_path),
            configuration_fingerprint(ValidationConfig(confidence=0.40), model_path),
        )
```

- [ ] **Step 2: Verify checkpoint tests fail**

Run: `python -m unittest -v test_pivc_validation.CheckpointTests`

Expected: FAIL because persistence and orchestration are missing.

- [ ] **Step 3: Implement atomic JSON persistence**

Write UTF-8 JSON to a sibling temporary file, flush and close it, then use `Path.replace()` to update `checkpoint.json`. Include schema version, timestamp, package versions, model path/hash, fixed configuration, and results keyed by stable `case_id`.

- [ ] **Step 4: Implement sequential CPU batch execution**

```python
def run_validation_batch(cases, model, config, output_dir, processor=process_validation_case):
    for index, case in enumerate(cases, start=1):
        if matching_complete_result_exists(case):
            continue
        result = processor(case, model, config, diagnostics_dir)
        update_checkpoint(result)
        print(f"[{index}/{len(cases)}] {case.case_id}: {result.status}")
```

Catch per-image exceptions and convert them to `REJECTED` with stage `unexpected_error`; do not catch `KeyboardInterrupt`. Save after every result.

- [ ] **Step 5: Run checkpoint and batch tests**

Run: `python -m unittest -v test_pivc_validation.CheckpointTests`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pi-cent_conv/pivc_validation.py pi-cent_conv/test_pivc_validation.py
git commit -m "feat: add resumable PIVC validation batch"
```

---

### Task 6: Thin Validation Notebook

**Files:**
- Create: `pi-cent_conv/run_pivc_validation.ipynb`
- Modify: `pi-cent_conv/test_pivc_validation.py`

**Interfaces:**
- Consumes: public interfaces from `pivc_validation.py`.
- Produces: an operator-facing notebook that runs or resumes the batch and prints compact status counts.

- [ ] **Step 1: Write a notebook structure test**

```python
def test_validation_notebook_contains_only_orchestration_calls(self):
    notebook = json.loads(Path("run_pivc_validation.ipynb").read_text("utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    self.assertIn("discover_validation_cases", source)
    self.assertIn("run_validation_batch", source)
    self.assertNotIn("def skeleton_to_graph", source)
```

- [ ] **Step 2: Verify the test fails because the notebook is absent**

Run: `python -m unittest -v test_pivc_validation.NotebookStructureTests`

Expected: FAIL with missing notebook.

- [ ] **Step 3: Create the notebook with these cells**

1. Imports and fixed paths.
2. Display the immutable `ValidationConfig`.
3. Discover cases and enforce the 40-image balance gate.
4. Load `best.pt` once on CPU.
5. Run/resume `run_validation_batch`.
6. Print counts for `ACCURATE`, `NOT ACCURATE`, and `REJECTED`.
7. Invoke the workbook builder only after all 40 cases have checkpoint records.

The notebook must not contain duplicated skeletonization, reconstruction, calibration, or workbook formatting logic.

- [ ] **Step 4: Run notebook structure and syntax checks**

Run: `python -m unittest -v test_pivc_validation.NotebookStructureTests`

Expected: PASS.

Run: `python -c "import json,ast,pathlib; n=json.loads(pathlib.Path('run_pivc_validation.ipynb').read_text('utf-8')); [ast.parse(''.join(c.get('source',[]))) for c in n['cells'] if c.get('cell_type')=='code']; print('OK')"`

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add pi-cent_conv/run_pivc_validation.ipynb pi-cent_conv/test_pivc_validation.py
git commit -m "feat: add PIVC validation notebook"
```

---

### Task 7: Excel Workbook Builder

**Files:**
- Create: `pi-cent_conv/build_validation_workbook.mjs`
- Create or modify: `pi-cent_conv/test_validation_workbook.mjs`

**Interfaces:**
- Consumes: `validation_outputs/checkpoint.json`, diagnostic image paths, output directory, and loader-provided artifact-tool runtime.
- Produces: `PIVC_Validation.xlsx`, plus temporary render previews used only for verification.

- [ ] **Step 1: Load the approved spreadsheet runtime and select a template**

Call `codex_app__load_workspace_dependencies` and create a Windows junction from the working directory's `node_modules` to the returned loader-provided `node_modules`. Open the spreadsheet template picker because no workbook template was supplied; continue with the default scientific-report styling if the picker is declined or unavailable.

- [ ] **Step 2: Mark workbook authoring exactly once**

Run from the spreadsheet skill directory:

```bash
node container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format xlsx
```

Expected: successful operation marker before the first workbook create/edit command.

- [ ] **Step 3: Write a failing builder contract test**

The test supplies a two-row fixture checkpoint and asserts that the exported workbook has exactly these sheets:

```text
Summary
Image Results
QC L35
QC L40
QC L45
QC L50
```

It also inspects `Image Results!A1:L3` and checks the twelve exact headers, formulas in I:L, and data validation values in F.

- [ ] **Step 4: Verify the contract test fails**

Run: `node test_validation_workbook.mjs`

Expected: FAIL because the builder is absent.

- [ ] **Step 5: Implement the workbook base and Image Results table**

```javascript
import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workbook = Workbook.create();
for (const name of ["Summary", "Image Results", "QC L35", "QC L40", "QC L45", "QC L50"]) {
  workbook.worksheets.add(name);
}
```

Write exactly these headers to `Image Results!A1:L1`:

```javascript
[
  "Filename", "Dressing condition", "Known length (cm)", "PIVC detected",
  "Marks detected", "Endpoints on PIVC line", "Corrected centreline length (px)",
  "Consecutive mark spacings (px)", "Pixels per centimetre",
  "Estimated PIVC length (cm)", "Absolute error (cm)", "Accuracy result"
]
```

Use formulas per data row:

```excel
I2 = checkpoint calibration value
J2 = IFERROR(G2/I2,"")
K2 = IF(J2="","",ABS(J2-C2))
L2 = IF(OR(D2<>"Yes",E2<2,G2="",I2=""),"REJECTED",IF(K2<=0.1,"ACCURATE","NOT ACCURATE"))
```

Set F-row validation values to `Unreviewed,Yes,No,N/A`. Use `Unreviewed` for any produced centreline and `N/A` otherwise. Format lengths and errors as `0.000`, pixels as `0.00`, and rates as `0.0%`. Freeze the header row and apply an Excel table with filters.

- [ ] **Step 6: Implement formula-driven Summary**

Create the eight length/dressing groups and one Overall row. Use bounded `COUNTIFS`, `COUNTIF`, `AVERAGEIFS`, `SUMPRODUCT`, and `MAXIFS` formulas referencing `'Image Results'!$A$2:$L$41`. Guard divisions with `IFERROR`.

Summary columns are exactly:

```text
Group, Total images, Measured images, Rejected images, Accurate images,
Inaccurate images, Measurement success rate, Accuracy among measured images,
Overall reliable-result rate, Mean absolute error (cm), RMSE (cm),
Maximum absolute error (cm)
```

Use green for accurate, amber for not accurate, red for rejected, and neutral blue/teal section headers. Do not rely on fill colour as the only status encoding.

- [ ] **Step 7: Embed diagnostic images into four QC sheets**

For each length sheet, place five with-dressing and five without-dressing panels in two clearly labeled sections. Use:

```javascript
sheet.images.add({
  dataUrl,
  anchor: { from: { row, col }, extent: { widthPx: 480, heightPx: 480 } },
});
```

Preserve aspect ratio by computing each extent from the diagnostic dimensions within a fixed panel box. Put filename, dressing condition, status, estimated length/error or rejection reason in cells above each image. Do not stretch images.

- [ ] **Step 8: Run workbook contract tests**

Run: `node test_validation_workbook.mjs`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add pi-cent_conv/build_validation_workbook.mjs pi-cent_conv/test_validation_workbook.mjs
git commit -m "feat: build PIVC validation workbook"
```

---

### Task 8: Complete 40-Image Validation Run

**Files:**
- Generate: `pi-cent_conv/validation_outputs/checkpoint.json`
- Generate: `pi-cent_conv/validation_outputs/diagnostics/*`
- Generate: `outputs/pivc_validation_<run_id>/PIVC_Validation.xlsx`

**Interfaces:**
- Consumes: all implemented modules, notebook orchestration, `best.pt`, and 40 validation images.
- Produces: the complete retained validation artifacts.

- [ ] **Step 1: Run the complete Python regression suite**

Run: `python -m unittest -v test_pivc_centerline.py test_pivc_validation.py`

Expected: all tests PASS with no unexpected warnings or errors.

- [ ] **Step 2: Run or resume the full CPU validation batch**

Execute `run_pivc_validation.ipynb` in the Python 3.11 environment or call its public orchestration functions from Python. Continue until checkpoint records exist for exactly 40 unique cases.

- [ ] **Step 3: Reconcile the checkpoint**

Run a validation command that asserts:

```python
assert len(results) == 40
assert Counter((r["length_group"], r["dressing_condition"]) for r in results) == expected_5_each
assert all(r["status"] in {"ACCURATE", "NOT ACCURATE", "REJECTED"} for r in results)
assert all(r["diagnostic_available"] == Path(r["diagnostic_path"]).is_file() for r in results)
```

- [ ] **Step 4: Generate the workbook from the final checkpoint**

Run: `node build_validation_workbook.mjs validation_outputs/checkpoint.json <output-directory>`

Expected: one `PIVC_Validation.xlsx` file.

- [ ] **Step 5: Inspect key workbook ranges**

Use artifact-tool:

```javascript
await workbook.inspect({
  kind: "table", range: "Image Results!A1:L41",
  include: "values,formulas", tableMaxRows: 8, tableMaxCols: 12,
});
await workbook.inspect({
  kind: "table", range: "Summary!A1:L11",
  include: "values,formulas", tableMaxRows: 11, tableMaxCols: 12,
});
```

Confirm 40 data rows, exact headers, correct known lengths/dressing conditions, and formula-derived results.

- [ ] **Step 6: Scan formula errors**

```javascript
await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
```

Expected: no formula errors.

- [ ] **Step 7: Render and visually inspect every sheet**

Render Summary, Image Results, QC L35, QC L40, QC L45, and QC L50. Verify titles, numbers, images, labels, and row/column sizing are readable and not clipped or overlapping. Make only focused layout corrections, then rerender affected sheets.

- [ ] **Step 8: Reconcile representative Excel rows with checkpoint values**

Check at least one accurate, one inaccurate, and one rejected case if those categories exist. Confirm filename, dressing, known length, mark count, corrected pixels, spacing, calibration, estimate, absolute error, status, and diagnostic mapping.

- [ ] **Step 9: Commit source code and tests only**

```bash
git add pi-cent_conv/pivc_validation.py pi-cent_conv/test_pivc_validation.py pi-cent_conv/run_pivc_validation.ipynb pi-cent_conv/build_validation_workbook.mjs pi-cent_conv/test_validation_workbook.mjs
git commit -m "feat: complete PIVC reliability validation workflow"
```

Do not commit model weights, raw validation images, generated diagnostics, checkpoints, preview images, or the final workbook unless the user explicitly requests repository retention.
