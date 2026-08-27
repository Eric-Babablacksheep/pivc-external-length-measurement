# PICC Measurement Post-Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-compatible local Python package that loads `best.pt`, extracts a minimal longest-path PICC centreline, calibrates it from projected mark centroids, and returns a measurement or explicit rejection with debug artifacts.

**Architecture:** Work inside `pi-cent_conv` with a small `picc_measurement` package and a thin command-line script. Pure geometry functions are separated from Ultralytics inference and tested with synthetic masks and paths. The first implementation uses an unsmoothed weighted skeleton geodesic and median adjacent mark spacing; robust missing-mark calibration is deferred.

**Tech Stack:** Python 3.11, Ultralytics, PyTorch CPU, NumPy, OpenCV, scikit-image, SciPy, NetworkX, Matplotlib, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-picc-postprocessing-design.md`

## Global Constraints

- Run inference on CPU by default.
- Resolve `picc` and `mark` IDs through `model.names`.
- Never use bounding-box dimensions for length measurement.
- Never use filename-derived ground truth as an algorithm input.
- Skeletonize only the cleaned predicted PICC mask.
- Use an 8-connected weighted longest endpoint-to-endpoint skeleton path.
- Use centreline path distance, not Euclidean centroid distance, for the 1 cm mark scale.
- Require at least two accepted marks; otherwise reject without a length.
- Keep the held-out test split out of development.
- Save both JSON decisions and a debug PNG.

---

### Task 1: Local package and test environment

**Files:**
- Create: `pi-cent_conv/requirements.txt`
- Create: `pi-cent_conv/picc_measurement/__init__.py`
- Create: `pi-cent_conv/tests/__init__.py`
- Create: `pi-cent_conv/tests/test_environment.py`

**Interfaces:**
- Produces: importable `picc_measurement` package and reproducible dependency list.

- [ ] **Step 1: Write the environment test**

```python
def test_package_imports():
    import cv2
    import networkx
    import numpy
    import scipy
    import skimage
    import torch
    import ultralytics

    assert torch.__version__
    assert ultralytics.__version__
```

- [ ] **Step 2: Run the test before installing dependencies**

Run from `pi-cent_conv`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pytest tests\test_environment.py -v
```

Expected: failure until pytest and runtime dependencies are installed.

- [ ] **Step 3: Add pinned-compatible dependency floors**

```text
ultralytics>=8.4,<9
torch>=2.2
numpy>=1.26
opencv-python>=4.9
scikit-image>=0.23
scipy>=1.12
networkx>=3.2
matplotlib>=3.8
pillow>=10
pytest>=8
```

- [ ] **Step 4: Install and verify**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests\test_environment.py -v
```

Expected: PASS.

### Task 2: Configuration, result types, and filename ground truth

**Files:**
- Create: `pi-cent_conv/picc_measurement/config.py`
- Create: `pi-cent_conv/picc_measurement/types.py`
- Create: `pi-cent_conv/picc_measurement/filename_labels.py`
- Create: `pi-cent_conv/tests/test_filename_labels.py`

**Interfaces:**
- Produces: `MeasurementConfig`, `Rejection`, `MeasurementResult`, and `parse_length_cm(path: Path) -> float | None`.

- [ ] **Step 1: Test filename parsing without algorithm leakage**

```python
from pathlib import Path
from picc_measurement.filename_labels import parse_length_cm


def test_known_prefixes():
    assert parse_length_cm(Path("L35_001.jpg")) == 3.5
    assert parse_length_cm(Path("L40_108.jpg")) == 4.0
    assert parse_length_cm(Path("L45_sample.png")) == 4.5
    assert parse_length_cm(Path("L50_001.jpeg")) == 5.0


def test_unknown_prefix_returns_none():
    assert parse_length_cm(Path("sample.jpg")) is None
```

- [ ] **Step 2: Implement the exact prefix map**

```python
PREFIX_TO_CM = {"L35": 3.5, "L40": 4.0, "L45": 4.5, "L50": 5.0}


def parse_length_cm(path):
    return PREFIX_TO_CM.get(path.stem.upper().split("_", 1)[0])
```

- [ ] **Step 3: Define immutable configuration and serializable result dataclasses**

Initial configuration fields include inference confidence 0.25, mask threshold 0.5, minimum PICC area 100 pixels, maximum mark-to-mask distance 20 pixels, minimum accepted marks 2, maximum adjacent-spacing coefficient of variation 0.25, maximum centreline/Euclidean ratio 1.8, and CPU device `cpu`. These are development defaults, not validated clinical thresholds.

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests\test_filename_labels.py -v
```

### Task 3: CPU model inference and original-resolution masks

**Files:**
- Create: `pi-cent_conv/picc_measurement/inference.py`
- Create: `pi-cent_conv/tests/test_inference_helpers.py`

**Interfaces:**
- Produces: `load_model(path: Path, device: str) -> YOLO`, `resolve_class_ids(names) -> dict[str, int]`, and `predict_instances(model, image_path, config) -> InferenceOutput`.

- [ ] **Step 1: Test class resolution**

```python
from picc_measurement.inference import resolve_class_ids


def test_resolves_classes_independent_of_order():
    assert resolve_class_ids({0: "mark", 1: "picc"}) == {"mark": 0, "picc": 1}
    assert resolve_class_ids({0: "picc", 1: "mark"}) == {"picc": 0, "mark": 1}
```

- [ ] **Step 2: Test missing or extra required names fail clearly**

Require both `picc` and `mark`; ignore unrelated extra names but reject a missing required name.

- [ ] **Step 3: Implement CPU loading and prediction extraction**

Call `model.predict(source=str(image_path), imgsz=960, conf=config.confidence, device=config.device, retina_masks=True, verbose=False)`. Resize each binary mask to the original image shape with nearest-neighbor interpolation only when shapes differ. Retain class name, confidence, mask, and bounding box; do not calculate length from boxes.

- [ ] **Step 4: Run helper tests**

```powershell
python -m pytest tests\test_inference_helpers.py -v
```

### Task 4: Minimal PICC-mask cleanup

**Files:**
- Create: `pi-cent_conv/picc_measurement/mask_processing.py`
- Create: `pi-cent_conv/tests/test_mask_processing.py`

**Interfaces:**
- Produces: `clean_picc_mask(mask, config) -> MaskCleanupResult`.

- [ ] **Step 1: Test removal of a small disconnected component**

Create a 100 x 100 binary mask with one long component and one 2 x 2 noise island; assert only the long component remains.

- [ ] **Step 2: Test largest-component retention does not change endpoints**

Create a horizontal 5-pixel-wide catheter mask and assert its bounding coordinates are unchanged after cleanup.

- [ ] **Step 3: Implement conservative cleanup**

Use connected-component labeling, retain the largest component above minimum area, and apply only a 3 x 3 elliptical morphological close when configured. Report original component count, retained area ratio, and whether the mask touches an image border. Reject an empty or undersized component.

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests\test_mask_processing.py -v
```

### Task 5: Weighted skeleton and longest ordered path

**Files:**
- Create: `pi-cent_conv/picc_measurement/centerline.py`
- Create: `pi-cent_conv/tests/test_centerline.py`

**Interfaces:**
- Produces: `extract_centerline(mask, config) -> CenterlineResult` with ordered `(row, column)` points, cumulative path distances, two unordered endpoints, geodesic length, Euclidean endpoint distance, branch count, and skeleton mask.

- [ ] **Step 1: Test exact straight-path length**

For a one-pixel horizontal skeleton containing pixels `(10, 3)` through `(10, 13)`, assert geodesic length 10 and endpoint distance 10.

- [ ] **Step 2: Test diagonal weights**

For pixels `(0, 0)`, `(1, 1)`, `(2, 2)`, assert length `2 * sqrt(2)`.

- [ ] **Step 3: Test longest path ignores a short spur**

Build a T-shaped synthetic skeleton whose main line is longer than its branch and assert the returned endpoints and path follow the main line.

- [ ] **Step 4: Implement skeleton graph extraction**

Use `skimage.morphology.skeletonize`, create an undirected NetworkX graph for foreground pixels, connect 8-neighbors with weight 1 or square-root 2, identify degree-1 endpoints, and find the maximum weighted shortest path over endpoint pairs. Reject fewer than two endpoints, disconnected skeletons, or a centreline/Euclidean ratio above configuration. Do not semantically name the endpoints and do not smooth the path.

- [ ] **Step 5: Run tests**

```powershell
python -m pytest tests\test_centerline.py -v
```

### Task 6: Mark centroids, projection, and simple calibration

**Files:**
- Create: `pi-cent_conv/picc_measurement/calibration.py`
- Create: `pi-cent_conv/tests/test_calibration.py`

**Interfaces:**
- Produces: `centroid(mask)`, `project_marks(mark_instances, cleaned_mask, centerline, config)`, and `calibrate(projected_marks, config) -> CalibrationResult`.

- [ ] **Step 1: Test centroid calculation**

Use a symmetric 3 x 3 mark mask centered at row 5, column 7 and assert centroid `(5.0, 7.0)`.

- [ ] **Step 2: Test projection onto ordered path**

Use a horizontal centreline with cumulative coordinates 0 through 20 and mark centroids near columns 5, 10, and 15; assert projected path coordinates 5, 10, and 15.

- [ ] **Step 3: Test median adjacent spacing**

For projected coordinates `[5, 15, 25]`, assert pixels per centimetre 10 and adjacent path distances `[10, 10]`.

- [ ] **Step 4: Test rejection cases**

Reject fewer than two marks and reject `[5, 15, 35]` when adjacent-spacing coefficient of variation exceeds the configured limit. Preserve each spacing in the rejection result so a missed mark is visible.

- [ ] **Step 5: Implement association and calibration**

Accept a mark only when its centroid is within the configured Euclidean distance of a cleaned-mask foreground pixel. Project it to the nearest centreline point, sort by cumulative path coordinate, remove duplicate projections, calculate adjacent path and Euclidean centroid distances, and select the median path spacing when consistency passes.

- [ ] **Step 6: Run tests**

```powershell
python -m pytest tests\test_calibration.py -v
```

### Task 7: Measurement orchestration and stable rejection behavior

**Files:**
- Create: `pi-cent_conv/picc_measurement/measurement.py`
- Create: `pi-cent_conv/tests/test_measurement.py`

**Interfaces:**
- Produces: `MeasurementEngine(model_path, config)` and `measure(image_path) -> MeasurementResult`.

- [ ] **Step 1: Test arithmetic independent of inference**

For centreline length 100 pixels and scale 20 pixels/cm, assert 5.0 cm.

- [ ] **Step 2: Test rejection propagation**

Mock inference with no PICC, multiple ambiguous PICCs, fewer than two marks, and inconsistent calibration; assert a stable code and `predicted_length_cm is None` for each.

- [ ] **Step 3: Implement the pipeline**

Resolve class IDs, infer instances, select exactly one plausible PICC, clean its mask, extract centreline, project marks, calibrate, and calculate `centreline_length_px / pixels_per_cm`. Parse filename ground truth only after prediction to calculate development absolute error. Store every intermediate diagnostic in the result.

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests\test_measurement.py -v
```

### Task 8: Debug visualization and command-line entry point

**Files:**
- Create: `pi-cent_conv/picc_measurement/visualization.py`
- Create: `pi-cent_conv/scripts/measure_image.py`
- Create: `pi-cent_conv/tests/test_cli.py`
- Preserve: `pi-cent_conv/load.py`

**Interfaces:**
- Consumes: `MeasurementResult`.
- Produces: `outputs/<image-stem>/result.json`, `debug.png`, and an exit code of 0 for valid measurement or 2 for a controlled rejection.

- [ ] **Step 1: Test CLI argument validation without loading a model**

Require `--model`, `--image`, and optional `--output-root`; reject missing paths with a concise message.

- [ ] **Step 2: Implement the debug figure**

Create panels for original image, raw PICC mask, cleaned mask, skeleton and ordered path, and original image with accepted/rejected mark centroids and projections. Include centreline pixels, Euclidean endpoint pixels, adjacent path spacings, pixels/cm, predicted length, ground truth, error, and rejection reasons in the title or text panel.

- [ ] **Step 3: Implement JSON serialization and CLI**

Run CPU inference once, create the per-image output directory, write JSON with NumPy values converted to built-in Python types, save `debug.png`, print a short result summary, and return the stable exit code.

- [ ] **Step 4: Run the full automated suite**

```powershell
python -m pytest -v
```

- [ ] **Step 5: Run the first real development image**

```powershell
python scripts\measure_image.py --model models\best.pt --image development_images\L35_001.jpg --output-root outputs
```

Expected: CPU inference completes and `outputs/L35_001/result.json` plus `debug.png` are created. The numerical result is not accepted as accurate until the overlay confirms mask coordinates, endpoints, ordered path, and mark projections.

### Task 9: First-image visual gate and configuration freeze candidate

**Files:**
- Modify only as evidence requires: `pi-cent_conv/picc_measurement/config.py`
- Modify only as evidence requires: the component responsible for a demonstrated failure.

**Interfaces:**
- Consumes: first real-image JSON and debug PNG.
- Produces: an evidence-backed development configuration candidate.

- [ ] **Step 1: Inspect the debug artifact**

Check that the raw and cleaned masks align with the original, the path spans only the visible catheter, endpoints match mask ends, no branch is included, genuine marks are accepted, printed text is rejected, and projected order follows the path.

- [ ] **Step 2: Compare the predicted length with the filename ground truth**

Record error without tuning on held-out test images. If invalid, use the saved intermediate values to identify the first failing stage rather than adjusting multiple thresholds.

- [ ] **Step 3: Re-run targeted unit and real-image checks after any change**

```powershell
python -m pytest -v
python scripts\measure_image.py --model models\best.pt --image development_images\L35_001.jpg --output-root outputs
```

- [ ] **Step 4: Stop before held-out validation**

Repeat development checks on non-test `L40`, `L45`, and `L50` examples only after the first image is geometrically correct. Do not open the test split in this stage.
