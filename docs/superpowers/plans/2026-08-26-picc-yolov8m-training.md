# PICC YOLOv8m Segmentation Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an executable Google Colab workflow that merges the four verified Roboflow segmentation exports, screens split leakage, trains `yolov8m-seg.pt`, persists artifacts, and downloads a verified `best.pt`.

**Architecture:** Sequential notebook sections consume the existing `dataset_roots` and `class_mappings` dictionaries created by the download/audit workflow. Preparation creates a collision-safe merged dataset in `/content/picc_dataset`; quality gates stop before training on structural or leakage failures. Training uses the validation split for early stopping, writes its run to Google Drive, and leaves the test split untouched for the later measurement-validation stage.

**Tech Stack:** Google Colab, Python 3, Ultralytics, PyTorch, PyYAML, Pillow, ImageHash, Google Drive.

**Spec:** `docs/superpowers/specs/2026-08-26-picc-yolov8-training-design.md`

## Global Constraints

- Use pretrained `yolov8m-seg.pt`.
- Preserve class ID `0 = mark` and class ID `1 = picc`.
- Preserve the existing Roboflow train/valid/test assignments.
- Train at image size 960 for at most 200 epochs with patience 30 and seed 42.
- Start with batch 4; retry with batch 2 only after a verified CUDA out-of-memory failure.
- Do not use the test split for training, early stopping, or hyperparameter selection.
- Do not perform pixel-to-centimetre conversion or claim measurement accuracy in this stage.
- Do not embed Roboflow API keys in notebook cells.

---

### Task 1: Runtime and source-dataset gate

**Files:**
- Create: `notebooks/picc_yolov8m_training.ipynb`

**Interfaces:**
- Consumes: `dataset_roots: dict[str, pathlib.Path]` and `class_mappings: dict[str, dict[int, str]]` from the completed download workflow.
- Produces: verified source-root dictionary and CUDA device report.

- [ ] **Step 1: Add a dependency and GPU setup cell**

```python
!pip install -q ultralytics pyyaml pillow imagehash

from pathlib import Path
import torch
import ultralytics

assert torch.cuda.is_available(), "Enable a GPU runtime before training."
print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
```

- [ ] **Step 2: Add a failing precondition check for missing source variables**

```python
required_lengths = {"L35", "L40", "L45", "L50"}
assert "dataset_roots" in globals(), "Rerun the four Roboflow download cells."
assert "class_mappings" in globals(), "Rerun the YAML inspection cells."
assert set(dataset_roots) == required_lengths
assert set(class_mappings) == required_lengths
```

- [ ] **Step 3: Validate roots and mappings**

```python
for length in sorted(required_lengths):
    root = Path(dataset_roots[length])
    assert root.is_dir(), f"Missing source dataset: {length} -> {root}"
    assert class_mappings[length] == {0: "mark", 1: "picc"}
```

- [ ] **Step 4: Run the cells and confirm the gate passes**

Expected: a CUDA GPU name is printed and all assertions pass.

### Task 2: Merge the four exports without changing splits

**Files:**
- Modify: `notebooks/picc_yolov8m_training.ipynb`
- Create at runtime: `/content/picc_dataset/data.yaml`
- Create at runtime: `/content/picc_dataset/merge_manifest.csv`

**Interfaces:**
- Consumes: verified source roots from Task 1.
- Produces: `MERGED_ROOT: pathlib.Path`, merged images/labels, and a manifest recording source and destination paths.

- [ ] **Step 1: Add merge helpers that pair files by stem and prefix destination names**

The helper must scan `train`, `valid`, and `test`, require identical image/label stem sets, copy each pair, and name the destination pair `{length}__{source_stem}{suffix}` and `{length}__{source_stem}.txt`.

- [ ] **Step 2: Add a clean-build guard**

Create `/content/picc_dataset` only when it does not exist. If it exists, stop with instructions to choose a clean runtime or explicitly remove only that exact directory before rebuilding; never silently merge into stale files.

- [ ] **Step 3: Write the merged YAML**

```yaml
path: /content/picc_dataset
train: images/train
val: images/valid
test: images/test
names:
  0: mark
  1: picc
```

- [ ] **Step 4: Verify expected counts**

```python
expected = {
    "train": {"L35": 86, "L40": 87, "L45": 89, "L50": 85},
    "valid": {"L35": 19, "L40": 19, "L45": 19, "L50": 18},
    "test":  {"L35": 18, "L40": 18, "L45": 19, "L50": 18},
}
assert sum(expected["train"].values()) == 347
assert sum(expected["valid"].values()) == 75
assert sum(expected["test"].values()) == 73
```

Note: the user-provided per-group figures sum to 347 training images and 495 total images. The earlier reported aggregate of 346 training images was an arithmetic inconsistency; the notebook must use file-derived counts and assert 347/75/73.

- [ ] **Step 5: Revalidate pairing and labels in the merged tree**

Expected: 495 image-label pairs, exactly one `picc` per image, no unknown class IDs, no five-token bounding-box rows, and no malformed or out-of-range polygon coordinates.

### Task 3: Screen cross-split leakage

**Files:**
- Modify: `notebooks/picc_yolov8m_training.ipynb`
- Create at runtime: `/content/picc_dataset/duplicate_report.csv`

**Interfaces:**
- Consumes: merged images from Task 2.
- Produces: exact-duplicate failures and near-duplicate candidates for inspection.

- [ ] **Step 1: Compute SHA-256 for every merged image**

Group identical hashes and fail if any group spans more than one split.

- [ ] **Step 2: Compute 64-bit perceptual hashes**

Compare pairs only across different splits. Report pairs with Hamming distance at most 4, including both paths, splits, lengths, and distance.

- [ ] **Step 3: Gate training**

Exact cross-split duplicates stop training. Near-duplicate candidates are displayed and saved for human inspection; the user must explicitly confirm they are acceptable or correct the split before continuing.

### Task 4: Persist metadata and configure the training run

**Files:**
- Modify: `notebooks/picc_yolov8m_training.ipynb`
- Create at runtime: Google Drive directory `MyDrive/PICC_AI/training_runs/yolov8m_seg_960_v1`

**Interfaces:**
- Consumes: leakage-approved merged dataset.
- Produces: mounted Drive path, environment record, and Ultralytics training configuration.

- [ ] **Step 1: Mount Google Drive and create the run parent**

Use `google.colab.drive.mount('/content/drive')` and place the run under `/content/drive/MyDrive/PICC_AI/training_runs`.

- [ ] **Step 2: Save reproducibility metadata**

Write Python, PyTorch, CUDA, Ultralytics, GPU, data-YAML hash, split counts, model name, and all hyperparameters to JSON before training.

- [ ] **Step 3: Instantiate the pretrained model**

```python
from ultralytics import YOLO
model = YOLO("yolov8m-seg.pt")
assert model.task == "segment"
```

### Task 5: Train and verify artifacts

**Files:**
- Modify: `notebooks/picc_yolov8m_training.ipynb`
- Create at runtime: Drive training artifacts including `weights/best.pt` and `weights/last.pt`.

**Interfaces:**
- Consumes: `model`, merged `data.yaml`, and Drive run path.
- Produces: trained model checkpoints and validation-training metrics.

- [ ] **Step 1: Start the approved training configuration**

```python
results = model.train(
    data="/content/picc_dataset/data.yaml",
    imgsz=960,
    epochs=200,
    batch=4,
    patience=30,
    pretrained=True,
    seed=42,
    deterministic=True,
    device=0,
    workers=2,
    degrees=10.0,
    translate=0.05,
    scale=0.10,
    hsv_h=0.01,
    hsv_s=0.15,
    hsv_v=0.15,
    fliplr=0.5,
    flipud=0.0,
    shear=0.0,
    perspective=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    save=True,
    save_period=10,
    plots=True,
    project="/content/drive/MyDrive/PICC_AI/training_runs",
    name="yolov8m_seg_960_v1",
    exist_ok=False,
)
```

- [ ] **Step 2: Handle only verified CUDA out-of-memory failures**

If the traceback explicitly reports CUDA out of memory, restart from a clean run name with `batch=2`. Do not catch unrelated exceptions or silently change parameters.

- [ ] **Step 3: Verify saved artifacts**

Assert that `weights/best.pt`, `weights/last.pt`, `args.yaml`, and `results.csv` exist and are non-empty. Load `best.pt` with `YOLO` and assert `task == "segment"` and `names == {0: "mark", 1: "picc"}`.

- [ ] **Step 4: Display validation-training results**

Display `results.csv`, `results.png`, confusion matrices, and validation-batch mask previews. State explicitly that these are segmentation results, not centimetre accuracy.

### Task 6: Recovery and model delivery

**Files:**
- Modify: `notebooks/picc_yolov8m_training.ipynb`
- Create at runtime: timestamped Drive backup and browser download of `best.pt`.

**Interfaces:**
- Consumes: verified `best.pt` and optional `last.pt`.
- Produces: recoverable run and downloaded checkpoint.

- [ ] **Step 1: Add an interruption-resume cell kept separate from normal training**

```python
from ultralytics import YOLO
resume_model = YOLO("/content/drive/MyDrive/PICC_AI/training_runs/yolov8m_seg_960_v1/weights/last.pt")
resume_model.train(resume=True)
```

- [ ] **Step 2: Create a timestamped backup**

Copy `best.pt`, `last.pt`, `results.csv`, `args.yaml`, metadata JSON, merged `data.yaml`, merge manifest, and duplicate report into a timestamped Drive folder.

- [ ] **Step 3: Download the selected model**

```python
from google.colab import files
files.download("/content/drive/MyDrive/PICC_AI/training_runs/yolov8m_seg_960_v1/weights/best.pt")
```

- [ ] **Step 4: Confirm the stage boundary**

Record the exact `best.pt` path and package versions for the next post-processing stage. Do not run the held-out test split or compute length metrics yet.
