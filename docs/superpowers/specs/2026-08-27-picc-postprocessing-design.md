# PICC Measurement Post-Processing Design

## Objective

Build a modular, CPU-compatible Python 3.11 measurement engine for Windows and VS Code. The engine loads the trained `best.pt` YOLOv8m instance-segmentation model, measures the visible curved PICC length in centimetres using graduation marks, reports explicit rejection reasons, and produces complete visual debugging output. It will be developed on non-test images whose filename prefixes encode manual lengths; held-out validation remains a later stage.

## Inputs and outputs

The initial command-line interface accepts one image and one model checkpoint. Development filenames use `L35`, `L40`, `L45`, or `L50` to represent 3.5, 4.0, 4.5, or 5.0 cm. The filename-derived value is development ground truth only and must never be used by the measurement algorithm.

For every image, the engine returns a structured result containing status, rejection reasons, predicted length when valid, filename-derived ground truth when available, absolute error for development reporting, model and configuration identifiers, and intermediate geometric measurements. It also saves an annotated debug image and machine-readable JSON.

## Package structure

```text
picc_measurement/
├── __init__.py
├── config.py
├── types.py
├── filename_labels.py
├── inference.py
├── mask_processing.py
├── centerline.py
├── calibration.py
├── measurement.py
└── visualization.py

scripts/
└── measure_image.py

models/
└── best.pt

development_images/
└── L35_001.jpg

outputs/
└── L35_001/
    ├── result.json
    └── debug.png

tests/
```

Each module has one responsibility and communicates through typed result objects. The measurement package must not depend on a web or mobile framework.

## Inference

Ultralytics loads `best.pt` on CPU and processes the original image without using filename length information. Class IDs are resolved from `model.names`; the code never assumes that `picc` or `mark` has a fixed numeric ID.

Predicted masks are mapped back to original-image dimensions before geometric processing. The inference result retains every candidate's class name, confidence, original-resolution binary mask, and bounding box for debugging only. Bounding boxes are not used for length measurement.

The engine selects one plausible PICC instance using confidence, mask area, continuity, and geometry. Ambiguous multiple-PICC predictions cause rejection rather than arbitrary selection. Mark candidates remain separate instances.

## PICC mask processing

Mask processing keeps the largest valid connected component, removes small isolated components, and closes only small gaps relative to catheter width. Thresholds are configuration values rather than unexplained constants. The cleaned mask must preserve endpoints and must not bridge distant structures.

Quality diagnostics include component count, retained-area ratio, endpoint contact with image borders, estimated mask width, and evidence of severe fragmentation. Cropping or ambiguous geometry causes rejection.

## Centreline extraction

Skeletonization operates only inside the cleaned PICC mask. The skeleton is converted into an 8-connected weighted graph with orthogonal edge weight 1 and diagonal edge weight square-root 2. Short spurs are pruned conservatively. The ordered centreline is the longest valid geodesic path between skeleton endpoints, not the sum of every skeleton pixel.

The ordered path is smoothed without materially shortening real curvature. Both raw geodesic and smoothed path lengths are retained; excessive disagreement causes rejection. The current two-class design can identify two geometric endpoints but cannot inherently name skin versus hub. Total external length does not require orientation, so endpoints remain unordered until later evidence supports reliable semantic naming.

## Mark processing

Each mark-mask centroid is checked against a configurable dilation zone around the cleaned PICC mask. Accepted centroids are projected onto the nearest ordered-centreline point and assigned a cumulative path coordinate. Duplicate projections and geometrically implausible detections are rejected. Accepted marks are sorted along the centreline.

At least two reliable marks are required. Images with fewer than two—including the known `L40_108` development case—return an explicit recapture/rejection result and no length.

## Calibration

The engine reports two calibration estimates:

1. A simple median of accepted adjacent projected-mark distances.
2. A robust one-dimensional lattice estimate of the underlying 1 cm spacing.

The robust estimator permits an observed gap to represent an integer multiple of the base 1 cm spacing, which protects against missed mark detections. It searches plausible base spacings, assigns adjacent gaps to positive integer multiples, minimizes robust residual error, and requires sufficient inlier support. This is preferred over assuming every detected adjacent pair is physically consecutive.

Calibration is rejected when the simple and robust estimates disagree beyond a configured tolerance, spacing residuals are inconsistent, the number of usable marks is insufficient, or the estimated scale is geometrically implausible. All accepted/rejected marks, gap distances, inferred gap multiples, residuals, and selected pixels-per-centimetre value are saved.

## Measurement and rejection

For a valid image:

```text
external_length_cm = smoothed_centreline_length_px / pixels_per_cm
```

The image is rejected rather than measured when inference, mask topology, endpoint visibility, centreline, mark association, or calibration fails a quality gate. Rejection reasons are stable codes with plain-language messages. Confidence thresholds and geometric tolerances are configuration parameters that will be finalized using development data and then frozen before held-out validation.

Filename-derived ground truth is parsed only after prediction and is used solely to report development error. It cannot influence detection, cleanup, calibration, or length calculation.

## Debugging output

The debug image contains panels for the original image, raw predicted PICC mask, cleaned mask, skeleton, ordered and smoothed centreline, unordered endpoints, all mark centroids, accepted/rejected marks, centreline projections, adjacent distances, inferred gap multiples, selected calibration, final length, and rejection reasons.

The JSON output stores the same numerical decisions so failures can be compared without relying only on visualization.

## Development and testing sequence

Implementation proceeds through unit-tested components: filename parsing, label-independent inference structures, mask cleanup, synthetic skeleton graphs, spur pruning, longest-path ordering, centroid projection, robust spacing estimation, measurement calculation, rejection behavior, and end-to-end debug generation. Synthetic masks and paths provide exact geometric expectations before real-image tuning.

The first real-image checkpoint runs a single non-test development image on CPU and verifies coordinate alignment and class resolution. Later checkpoints add geometry one component at a time. The held-out test split is not opened or tuned against during this stage.

## Stage boundary

This stage ends when the engine can load `best.pt`, process development images, return valid measurements or explicit rejection reasons, and save complete debug artifacts. It does not establish clinical performance. The following validation stage will freeze configuration and evaluate mask overlap, mark precision/recall, measurement success, rejection behavior, centimetre error, bias, RMSE, repeatability, change error, and movement-threshold sensitivity/specificity on held-out data.

## Known limitations

- The accepted Roboflow split contains perceptual-similarity candidates, so future results must disclose possible leakage.
- Validation showed substantially weaker mark-mask performance than PICC-mask performance; calibration failures and false marks are expected development risks.
- Per-image mark calibration cannot succeed with fewer than two reliable marks.
- Two segmentation classes do not semantically identify the skin and hub endpoints.
- CPU inference is supported, but it will be slower than CUDA inference.
