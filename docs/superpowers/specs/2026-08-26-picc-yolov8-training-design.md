# PICC YOLOv8 Segmentation Training Design

## Objective

Build a reproducible Google Colab workflow that combines the four verified Roboflow instance-segmentation exports, trains a two-class YOLOv8 segmentation model, preserves training artifacts, and produces a downloadable `best.pt` checkpoint. Pixel-to-centimetre conversion and complete measurement validation are explicitly outside this stage.

## Confirmed inputs

- Four independently exported Roboflow datasets: `L35`, `L40`, `L45`, and `L50`.
- 495 images total, currently split into 347 training, 75 validation, and 73 testing images.
- Every export uses the same mapping: class `0` is `mark`; class `1` is `picc`.
- Labels are YOLO polygon segmentation labels, with no detected bounding-box-only or malformed rows.
- Images have length-prefixed filenames but no capture-session or continuous-burst identifiers.
- Roboflow preprocessing used auto-orientation, 960 x 960 aspect-ratio-preserving fit, and no augmentation.

## Dataset preparation

The notebook will preserve each export's existing train, validation, and test assignment. It will merge the exports into one Ultralytics dataset with `images/{train,valid,test}` and `labels/{train,valid,test}`. Every destination filename will be prefixed with its length group to prevent collisions, even if Roboflow preserved the original prefix.

Before training, the workflow will enforce these gates:

- Exactly one matching label file per image and no orphan labels.
- Exactly one `picc` instance per image. Any violation stops the workflow for correction in Roboflow and re-export.
- Polygon rows use only known class IDs and normalized coordinate pairs.
- Split totals and per-length counts match the source exports.
- Exact duplicate files do not occur across splits.
- Likely near-duplicates across splits are reported using perceptual hashing for human inspection. Because no session metadata exists, this screening reduces but cannot eliminate leakage risk.

Images with fewer than two genuine `mark` instances may remain in segmentation training if their annotations are correct. They will later be rejected by the measurement engine and excluded from successful-measurement calculations as appropriate.

The merged `data.yaml` will declare:

```yaml
path: /content/picc_dataset
train: images/train
val: images/valid
test: images/test
names:
  0: mark
  1: picc
```

## Training configuration

Training will use the Ultralytics Python API and pretrained `yolov8m-seg.pt` weights on a Colab GPU. The initial run will use:

- Image size: 960.
- Epoch limit: 200.
- Batch size: 4, with an explicit reduction to 2 only if a verified CUDA out-of-memory error occurs.
- Early-stopping patience: 30.
- Seed: 42 and deterministic mode where supported.
- Rotation: 10 degrees.
- Translation: 0.05.
- Scale: 0.10.
- HSV adjustments: `hsv_h=0.01`, `hsv_s=0.15`, `hsv_v=0.15`.
- Horizontal flip probability: 0.5; vertical flip disabled.
- Shear, perspective, Mosaic, MixUp, and Copy-Paste disabled.
- Pretrained weights enabled.
- Periodic checkpoints saved every 10 epochs.

The workflow will confirm CUDA availability and report the assigned GPU before training. Training will stop instead of silently running a 200-epoch job on CPU.

## Artifact persistence and interruption recovery

Google Drive will be mounted before training. Ultralytics run outputs will be written to a dedicated Drive directory so that `best.pt`, `last.pt`, `results.csv`, plots, and training arguments survive Colab runtime loss. If training is interrupted, a separate resume cell will load `last.pt` with `resume=True`; it will not start a new experiment inadvertently.

## Model selection and stage boundary

Ultralytics will use the validation split during training for early stopping and checkpoint selection. The training-stage report will include box and mask precision, recall, mAP50, and mAP50-95 from the validation process, while clearly stating that these metrics do not establish centimetre-measurement accuracy.

The held-out test split will not be used for hyperparameter selection or training-stage decisions. Its complete evaluation belongs to the later validation stage after centreline extraction, graduation-mark calibration, image-quality rejection, and length-change calculation are implemented.

At the end of this stage, the notebook will verify that `best.pt` exists, reload it as a segmentation model, copy a timestamped backup to Drive, and initiate a browser download of `best.pt`. The retained deliverables are:

- `best.pt` as the selected model for post-processing development.
- `last.pt` for interrupted-run recovery.
- Training arguments and package-version record.
- `results.csv` and generated training/validation plots.
- The merged dataset manifest and duplicate-screening report.

## Limitations

- Filename-only grouping cannot prove independence between splits.
- Perceptual-hash screening will not identify every related capture.
- Training and segmentation mAP do not validate external PICC length measurement.
- The proof-of-concept movement threshold will not be assessed until the later complete-pipeline validation stage.
- At 960 px, `yolov8m-seg.pt` may exceed the memory available on some free Colab GPUs at batch size 4; a verified CUDA out-of-memory error will require restarting with batch size 2.
