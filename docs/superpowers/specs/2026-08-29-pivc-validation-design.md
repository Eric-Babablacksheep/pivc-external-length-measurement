# PIVC Measurement Validation Design

## Objective

Build a reproducible local Python validation workflow that processes 40 held-out phantom images, evaluates the reliability of the existing PIVC segmentation, centreline reconstruction, mark-based pixel calibration, and centimetre conversion logic, and produces an auditable Excel workbook with image-level results, grouped summary statistics, and diagnostic overlays.

This stage evaluates the algorithm and post-processing logic. It does not establish clinical validity.

## Confirmed validation dataset

The validation root is `pi-cent_conv/test_images`. It contains four known external-length groups:

- `L35`: 3.5 cm.
- `L40`: 4.0 cm.
- `L45`: 4.5 cm.
- `L50`: 5.0 cm.

Each length folder contains:

- `with_dressing`: five images.
- `without_dressing`: five images.

The complete validation set therefore contains 40 images, balanced across length and dressing condition. Folder names, rather than filename patterns or image orientation, define the known length and dressing condition.

## Fixed inference and measurement configuration

Every image must use the same trained `best.pt` segmentation model and the same configuration. The initial batch uses:

- Device: CPU.
- Inference image size: 960.
- Confidence threshold: 0.50.
- IoU threshold: 0.70.
- Retina masks enabled.
- Class mapping: class `0` is `mark`; class `1` remains named `picc` in the trained model but represents the PIVC.

No threshold may be changed for an individual validation image. Configuration values must be included in the retained validation metadata.

## Batch-processing architecture

A reusable Python batch-validation module will perform image processing, while a small Jupyter notebook will configure and start the run, display progress, resume interrupted work, and initiate workbook generation.

For each image, the pipeline will:

1. Load the image and run instance-segmentation inference.
2. identify the usable PIVC prediction and detected mark predictions.
3. Separate significant PIVC mask components.
4. Skeletonize component paths.
5. Build a rotation-independent safe component chain.
6. Exclude unreliable narrow fragments.
7. Repair disconnected and mark-associated centreline defects.
8. Project mark centroids onto the corrected centreline.
9. Use consecutive mark spacing, confirmed as 1.0 cm, to calculate pixels per centimetre.
10. Convert the complete corrected centreline length to centimetres.
11. Compare the estimate with the known folder-defined length.
12. Save a structured result and diagnostic overlay.

Processing continues after an individual rejection. Results are checkpointed after every image so a CPU interruption can resume without repeating completed images. A resumed run must skip only images whose result and diagnostic metadata are complete for the current configuration.

## Measurement and accuracy definitions

For a valid measurement:

```text
signed_error_cm = estimated_length_cm - known_length_cm
absolute_error_cm = abs(signed_error_cm)
```

The confirmed accuracy tolerance is 0.10 cm:

- `ACCURATE`: a valid measurement with absolute error less than or equal to 0.10 cm.
- `NOT ACCURATE`: a valid measurement with absolute error greater than 0.10 cm.
- `REJECTED`: the pipeline cannot safely produce a measurement.

Rejection conditions include no usable PIVC, fewer than two usable marks, an unsafe component chain, inconsistent mark calibration, or any other measurement safety-gate failure. Rejected images remain in the denominator for overall reliability statistics.

Two distinct rates must be reported:

```text
measurement_success_rate = measured_images / total_images
accuracy_among_measured = accurate_images / measured_images
overall_reliable_result_rate = accurate_images / total_images
```

This prevents rejected images from being hidden by accuracy calculations limited to successfully measured cases.

## Diagnostic overlay

Each image will receive a saved diagnostic overlay whenever enough processing state exists. A successful overlay will show:

- the PIVC mask with transparency;
- the final corrected centreline;
- reconstructed bridges and mark-repaired sections in contrasting colours;
- detected and projected mark centres;
- a highlighted centreline interval between each consecutive mark pair;
- a label such as `301.87 px = 1.00 cm`;
- both terminal centreline endpoints;
- known length, estimated length, absolute error, and automatic status.

For rejected images, the overlay will show the last valid processing stage and the rejection reason when possible. Diagnostic filenames will uniquely map to image-result rows.

## Excel workbook

The final workbook contains six sheets.

### Summary

Rows cover each combination of length and dressing condition plus an overall row. The sheet reports:

- group;
- total images;
- measured images;
- rejected images;
- accurate images;
- inaccurate images;
- measurement success rate;
- accuracy among measured images;
- overall reliable-result rate;
- mean absolute error;
- RMSE;
- maximum absolute error.

Summary calculations must be formula-driven from the `Image Results` sheet and must handle empty measured groups without formula errors.

### Image Results

This sheet contains exactly these twelve columns:

1. Filename.
2. Dressing condition.
3. Known length (cm).
4. PIVC detected.
5. Marks detected.
6. Endpoints on PIVC line.
7. Corrected centreline length (px).
8. Consecutive mark spacings (px).
9. Pixels per centimetre.
10. Estimated PIVC length (cm).
11. Absolute error (cm).
12. Accuracy result.

`Endpoints on PIVC line` is a manual visual-review field with the allowed values:

- `Unreviewed`: default for a produced diagnostic image.
- `Yes`: both plotted terminal endpoints visibly correspond to the physical external PIVC endpoints.
- `No`: either endpoint terminates early, extends into the hub, or lies away from the PIVC.
- `N/A`: no centreline was produced.

Known length, estimated length, and error remain typed numeric values. Pixels per centimetre, estimated length, absolute error, and accuracy classification are auditable formula-driven values where practical. Rejected rows leave unavailable numeric measurements blank and use `REJECTED` as the accuracy result.

### Visual QC sheets

The remaining sheets are:

- `QC L35`.
- `QC L40`.
- `QC L45`.
- `QC L50`.

Each sheet contains ten diagnostic panels, clearly separated into five with-dressing and five without-dressing cases. Every panel includes the filename, automatic result, estimated length or rejection reason, and the diagnostic overlay. Images must be resized for legibility and reasonable workbook size without changing their aspect ratio.

## Reliability interpretation

The workbook evaluates several separate properties:

- Pipeline coverage: whether the algorithm produces a safe measurement.
- Numerical accuracy: whether a produced estimate is within 0.10 cm.
- Dressing robustness: differences between with- and without-dressing groups.
- Length robustness: differences across 3.5, 4.0, 4.5, and 5.0 cm groups.
- Visual plausibility: whether a human reviewer confirms both endpoints lie on the PIVC.

An apparently accurate numerical result with incorrect endpoints remains visible through the manual endpoint-review field and must not be interpreted as geometrically reliable.

## Error handling and retained artifacts

The validation run retains:

- a machine-readable checkpoint containing one record per attempted image;
- one diagnostic image per case when possible;
- the fixed run configuration and package versions;
- the final Excel workbook.

An individual exception is converted into a rejected result with a concise stage and reason in the checkpoint. The batch must not silently omit unreadable or rejected files.

## Verification requirements

Before delivery:

- Confirm exactly 40 expected images and the balanced 5/5 folder structure.
- Unit-test folder metadata parsing, accuracy classification, rejection handling, resume behavior, and summary calculations.
- Run the complete batch with one fixed configuration.
- Reconcile workbook group counts with checkpoint counts.
- Scan workbook formulas for errors.
- Inspect representative numeric rows against checkpoint values.
- Render and visually inspect every workbook sheet for readable text, unclipped values, and correctly placed diagnostic images.

## Limitations

- The validation set contains phantom images and does not establish clinical performance.
- Folder-defined nominal lengths are treated as ground truth for this stage.
- The dataset is small, so subgroup statistics are descriptive rather than definitive.
- Manual endpoint review introduces reviewer judgment but is intentionally retained to detect geometrically incorrect yet numerically plausible results.
- The 0.10 cm threshold is the project-defined accuracy rule for this validation stage and is not presented as a clinical standard.
