"""Dataset metadata and structured results for PIVC validation images."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from pivc_centerline import (
    calibrate_length_from_marks,
    reconstruct_continuous_centerline,
    repair_centerline_through_marks,
)


@dataclass(frozen=True)
class ValidationConfig:
    """Inference and measurement settings for the v2 workflow."""

    imgsz: int = 960

    # PIVC instances must meet the user-selected threshold.
    confidence: float = 0.50

    # Marks may be smaller and less confident than the PIVC.
    mark_confidence: float = 0.30

    iou: float = 0.70
    device: str = "cpu"

    known_mark_spacing_cm: float = 1.0
    accuracy_tolerance_cm: float = 0.10

    # Post-process at a smaller resolution to reduce skeletonisation time.
    # All masks, paths and marks are scaled together, so pixel/cm conversion
    # remains internally consistent.
    processing_max_dimension: int = 1280


@dataclass(frozen=True)
class ValidationCase:
    """One validation image and metadata encoded by its folder names."""

    image_path: Path
    length_group: str
    known_length_cm: float
    dressing_condition: str


@dataclass
class ValidationResult:
    """JSON-safe record for one attempted validation image.

    The fields intentionally contain only JSON-compatible scalar values and
    lists.  This allows a result to be passed directly through
    :func:`dataclasses.asdict` before checkpoint serialization.
    """

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
    diagnostic_error: str = ""
    config_fingerprint: str = ""


@dataclass
class SegmentationOutput:
    """Model-independent masks needed by one PIVC measurement attempt.

    Array coordinates are always ``(row, column)`` in the original image
    coordinate system.  Keeping prediction adaptation separate from geometry
    makes the safety gates unit-testable without loading a YOLO model.
    """

    image_rgb: np.ndarray
    pivc_confidence: float
    pivc_mask: np.ndarray
    mark_masks: list[np.ndarray]

def resize_segmentation_output(
    output: SegmentationOutput,
    maximum_dimension: int,
) -> SegmentationOutput:
    """
    Resize the image and all masks to one common processing resolution.

    Because the PIVC centreline and mark spacing are scaled together,
    their pixel ratio remains approximately invariant.
    """

    image_rgb = np.asarray(output.image_rgb)
    pivc_mask = np.asarray(output.pivc_mask, dtype=np.uint8)
    mark_masks = [
        np.asarray(mask, dtype=np.uint8)
        for mask in output.mark_masks
    ]

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise RuntimeError("The segmentation image must be an RGB image.")

    height, width = image_rgb.shape[:2]
    largest_dimension = max(height, width)

    if maximum_dimension <= 0:
        raise ValueError("processing_max_dimension must be positive.")

    if largest_dimension <= maximum_dimension:
        return output

    scale = maximum_dimension / float(largest_dimension)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_shape = (resized_width, resized_height)

    resized_image = cv2.resize(
        image_rgb,
        resized_shape,
        interpolation=cv2.INTER_AREA,
    )

    resized_pivc = cv2.resize(
        pivc_mask,
        resized_shape,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    resized_marks = [
        cv2.resize(
            mask,
            resized_shape,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        for mask in mark_masks
    ]

    return SegmentationOutput(
        image_rgb=resized_image,
        pivc_confidence=output.pivc_confidence,
        pivc_mask=resized_pivc,
        mark_masks=resized_marks,
    )

class SegmentationPredictionError(RuntimeError):
    """A prediction-stage error that can become a structured rejection."""


LENGTHS_CM = {"L35": 3.5, "L40": 4.0, "L45": 4.5, "L50": 5.0}
DRESSING_GROUPS = ("with_dressing", "without_dressing")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _unknown_folder_error(folder: Path, expected: Sequence[str]) -> ValueError:
    expected_text = ", ".join(expected)
    return ValueError(
        f"Unknown folder '{folder.name}' under '{folder.parent}'; "
        f"expected one of: {expected_text}."
    )


def discover_validation_cases(root: Path) -> list[ValidationCase]:
    """Discover images from ``length/dressing/image`` validation folders.

    Only the approved length and dressing folder names are accepted. Images
    are returned in deterministic length-group, dressing-group, and filename
    order. Non-image files are ignored so that ordinary directory metadata
    (for example, thumbnail files) does not become a validation case.
    """

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Validation image root is not a directory: {root}")

    expected_lengths = tuple(sorted(LENGTHS_CM))
    for entry in root.iterdir():
        if entry.is_dir() and entry.name not in LENGTHS_CM:
            raise _unknown_folder_error(entry, expected_lengths)

    cases: list[ValidationCase] = []
    resolved_paths: set[Path] = set()

    for length_group in expected_lengths:
        length_dir = root / length_group
        if not length_dir.is_dir():
            continue

        for entry in length_dir.iterdir():
            if entry.is_dir() and entry.name not in DRESSING_GROUPS:
                raise _unknown_folder_error(entry, DRESSING_GROUPS)

        for dressing_condition in DRESSING_GROUPS:
            dressing_dir = length_dir / dressing_condition
            if not dressing_dir.is_dir():
                continue

            for entry in dressing_dir.iterdir():
                if entry.is_dir():
                    raise ValueError(
                        f"Unknown folder '{entry.name}' under '{dressing_dir}'."
                    )

            images = sorted(
                (
                    entry
                    for entry in dressing_dir.iterdir()
                    if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=lambda path: path.name,
            )
            for image_path in images:
                resolved_path = image_path.resolve()
                if resolved_path in resolved_paths:
                    raise ValueError(
                        f"Duplicate resolved image path discovered: {resolved_path}"
                    )
                resolved_paths.add(resolved_path)
                cases.append(
                    ValidationCase(
                        image_path=image_path,
                        length_group=length_group,
                        known_length_cm=LENGTHS_CM[length_group],
                        dressing_condition=dressing_condition,
                    )
                )

    return cases


def validate_case_balance(cases: Sequence[ValidationCase]) -> None:
    """Require exactly five images in every approved validation subgroup."""

    expected_subgroups = {
        (length_group, dressing_condition)
        for length_group in LENGTHS_CM
        for dressing_condition in DRESSING_GROUPS
    }
    counts = Counter((case.length_group, case.dressing_condition) for case in cases)

    unknown_subgroups = set(counts) - expected_subgroups
    if unknown_subgroups:
        subgroup = min(unknown_subgroups)
        raise RuntimeError(f"Unknown validation subgroup: {subgroup[0]}/{subgroup[1]}")

    for subgroup in sorted(expected_subgroups):
        count = counts[subgroup]
        if count != 5:
            raise RuntimeError(
                f"Validation subgroup {subgroup[0]}/{subgroup[1]} has {count} "
                "image(s); expected 5."
            )


def classify_accuracy(
    estimated_cm: float, known_cm: float, tolerance_cm: float
) -> tuple[str, float]:
    """Classify a measured length using the inclusive absolute-error limit.

    Decimal conversion from the string representation avoids a binary
    floating-point artifact changing an error mathematically equal to the
    tolerance from ``ACCURATE`` to ``NOT ACCURATE``.
    """

    values = {
        "estimated_cm": estimated_cm,
        "known_cm": known_cm,
        "tolerance_cm": tolerance_cm,
    }
    if any(not isinstance(value, (int, float)) for value in values.values()):
        raise TypeError("accuracy values must be numeric")
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("accuracy values must be finite")
    if tolerance_cm < 0:
        raise ValueError("tolerance_cm must be non-negative")

    try:
        error_decimal = abs(Decimal(str(estimated_cm)) - Decimal(str(known_cm)))
        tolerance_decimal = Decimal(str(tolerance_cm))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("accuracy values must be valid decimals") from exc

    error = float(error_decimal)
    status = "ACCURATE" if error_decimal <= tolerance_decimal else "NOT ACCURATE"
    return status, error


def rejected_result(
    case: ValidationCase, stage: str, reason: str
) -> ValidationResult:
    """Create a rejected result while retaining the case and failure context."""

    if not stage:
        raise ValueError("rejection stage must not be empty")
    if not reason:
        raise ValueError("rejection reason must not be empty")

    case_id = "/".join(
        (case.length_group, case.dressing_condition, case.image_path.name)
    )
    return ValidationResult(
        case_id=case_id,
        filename=case.image_path.name,
        image_path=str(case.image_path),
        length_group=case.length_group,
        dressing_condition=case.dressing_condition,
        known_length_cm=float(case.known_length_cm),
        rejection_stage=str(stage),
        rejection_reason=str(reason),
    )


def _normalise_binary_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean mask aligned to the source-image shape."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise SegmentationPredictionError("A predicted instance mask is not 2-dimensional.")
    if array.shape != target_shape:
        array = cv2.resize(
            array.astype(np.float32),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return array > 0.5


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def _to_numpy(value) -> np.ndarray:
    """Detach common tensor containers without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def predict_segmentation(
    model, image_path: Path, config: ValidationConfig
) -> SegmentationOutput:
    """Adapt one fixed-setting Ultralytics segmentation prediction.

    A duplicate PIVC prediction is ignored only when it substantially overlaps
    the strongest PIVC mask. Distinct PIVC predictions are unsafe ambiguity,
    rather than an opportunity to concatenate arbitrary objects.
    """

    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SegmentationPredictionError(f"Unreadable image: {image_path}")
    if getattr(model, "task", None) != "segment":
        raise SegmentationPredictionError("Loaded model task must be 'segment'.")

    names = getattr(model, "names", None)
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not {"mark", "picc"}.issubset(
        {str(name) for name in names.values()}
    ):
        raise SegmentationPredictionError(
            "Segmentation model must define both 'mark' and 'picc' classes."
        )

    try:
        inference_confidence = min(
            float(config.confidence),
            float(config.mark_confidence),
        )

        prediction = model.predict(
            source=str(image_path),
            imgsz=config.imgsz,
            conf=inference_confidence,
            iou=config.iou,
            device=config.device,
            retina_masks=True,
            max_det=30,
            verbose=False,
        )[0]
    except Exception as exc:  # ultralytics errors must not terminate a batch
        raise SegmentationPredictionError(f"Model prediction failed: {exc}") from exc

    masks = getattr(prediction, "masks", None)
    boxes = getattr(prediction, "boxes", None)
    if masks is None or boxes is None or getattr(masks, "data", None) is None:
        raise SegmentationPredictionError("Prediction contains no instance masks.")

    mask_data = _to_numpy(masks.data)
    class_ids = _to_numpy(getattr(boxes, "cls", []))
    confidences = _to_numpy(getattr(boxes, "conf", []))
    if (
        mask_data.ndim != 3
        or len(mask_data) != len(class_ids)
        or len(mask_data) != len(confidences)
    ):
        raise SegmentationPredictionError("Prediction mask and box counts do not match.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    class_names = getattr(prediction, "names", names)
    if isinstance(class_names, (list, tuple)):
        class_names = dict(enumerate(class_names))
    if not isinstance(class_names, dict):
        class_names = names

    pivc_candidates: list[tuple[float, np.ndarray]] = []
    mark_masks: list[np.ndarray] = []
    for raw_mask, class_id, confidence in zip(mask_data, class_ids, confidences):
        class_name = str(class_names.get(int(class_id), ""))
        binary_mask = _normalise_binary_mask(raw_mask, image_rgb.shape[:2])
        if not binary_mask.any():
            continue
        instance_confidence = float(confidence)

        if class_name == "picc":
            if instance_confidence >= config.confidence:
                pivc_candidates.append(
                    (instance_confidence, binary_mask)
                )

        elif class_name == "mark":
            if instance_confidence >= config.mark_confidence:
                mark_masks.append(binary_mask)

    if not pivc_candidates:
        raise SegmentationPredictionError("No usable PIVC mask was detected.")

    pivc_candidates.sort(key=lambda item: item[0], reverse=True)
    primary_confidence, primary_mask = pivc_candidates[0]
    for _, candidate_mask in pivc_candidates[1:]:
        if _mask_iou(primary_mask, candidate_mask) < 0.10:
            raise SegmentationPredictionError(
                "Multiple non-overlapping PIVC masks were detected."
            )

    return SegmentationOutput(
        image_rgb=image_rgb,
        pivc_confidence=primary_confidence,
        pivc_mask=primary_mask,
        mark_masks=mark_masks,
    )


def _skeleton_adjacency(skeleton: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    coordinates = np.argwhere(skeleton)
    index_by_coordinate = {tuple(point): index for index, point in enumerate(coordinates)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in coordinates]
    for index, (row, column) in enumerate(coordinates):
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                neighbour = index_by_coordinate.get(
                    (int(row + row_delta), int(column + column_delta))
                )
                if neighbour is not None:
                    adjacency[index].append(
                        (neighbour, math.hypot(row_delta, column_delta))
                    )
    return coordinates, adjacency


def _shortest_paths(
    adjacency: list[list[tuple[int, float]]], start: int
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(len(adjacency), np.inf, dtype=np.float64)
    previous = np.full(len(adjacency), -1, dtype=np.int32)
    distances[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        for neighbour, weight in adjacency[current]:
            candidate = distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = current
                heapq.heappush(queue, (candidate, neighbour))
    return distances, previous


def _longest_skeleton_path(component_mask: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(component_mask.astype(bool))
    coordinates, adjacency = _skeleton_adjacency(skeleton)
    if len(coordinates) < 2:
        raise RuntimeError("A significant PIVC component has no usable skeleton path.")

    endpoints = [index for index, neighbours in enumerate(adjacency) if len(neighbours) == 1]
    candidates = endpoints if len(endpoints) >= 2 else list(range(len(coordinates)))
    best_distance = -1.0
    best_start = best_end = None
    best_previous = None
    for start in candidates:
        distances, previous = _shortest_paths(adjacency, start)
        for end in candidates:
            if end == start or not math.isfinite(float(distances[end])):
                continue
            if float(distances[end]) > best_distance:
                best_distance = float(distances[end])
                best_start, best_end, best_previous = start, end, previous
    if best_start is None or best_previous is None:
        raise RuntimeError("A significant PIVC component has no endpoint-to-endpoint path.")

    indices = [best_end]
    while indices[-1] != best_start:
        predecessor = int(best_previous[indices[-1]])
        if predecessor < 0:
            raise RuntimeError("PIVC skeleton path reconstruction failed.")
        indices.append(predecessor)
    indices.reverse()
    return coordinates[indices].astype(np.float64)


def component_paths_from_mask(mask: np.ndarray) -> list[dict]:
    """Extract significant PIVC components as longest centreline paths.

    Components are intentionally not bridged here: gap reasoning remains the
    conservative, orientation-independent responsibility of
    :func:`reconstruct_continuous_centerline`.
    """

    binary_mask = np.asarray(mask).astype(bool)
    if binary_mask.ndim != 2 or not binary_mask.any():
        return []
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    areas = statistics[1:, cv2.CC_STAT_AREA]
    if component_count <= 1 or len(areas) == 0:
        return []
    area_threshold = max(50, int(0.01 * int(areas.max())))
    components: list[dict] = []
    for label in range(1, component_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < area_threshold:
            continue
        component_mask = labels == label
        try:
            path = _longest_skeleton_path(component_mask)
        except RuntimeError:
            continue
        widths = 2.0 * distance_transform_edt(component_mask)[
            path[:, 0].astype(int), path[:, 1].astype(int)
        ]
        if len(widths) == 0 or float(np.median(widths)) <= 0:
            continue
        components.append(
            {
                "component_id": label - 1,
                "path_array": path,
                "median_width": float(np.median(widths)),
                "area": area,
            }
        )
    return components


def _mark_centres(mark_masks: Sequence[np.ndarray]) -> list[tuple[float, float]]:
    centres: list[tuple[float, float]] = []
    for mask in mark_masks:
        rows, columns = np.nonzero(np.asarray(mask).astype(bool))
        if len(rows):
            centres.append((float(rows.mean()), float(columns.mean())))
    return centres


def _rgb_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """Validate and normalise a source image for local colour evidence."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError("Hub-side endpoint is ambiguous: source image is not RGB.")
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    if image.size == 0 or not np.isfinite(image).all():
        raise RuntimeError("Hub-side endpoint is ambiguous: source image is invalid.")
    if float(image.max()) <= 1.0:
        image *= 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _endpoint_blue_hub_score(
    hsv_image: np.ndarray, endpoint: np.ndarray, radius: int
) -> float:
    """Measure saturated blue hub material in a physical-endpoint neighbourhood."""

    height, width = hsv_image.shape[:2]
    row = int(round(float(endpoint[0])))
    column = int(round(float(endpoint[1])))
    row_start, row_end = max(0, row - radius), min(height, row + radius + 1)
    column_start, column_end = max(0, column - radius), min(width, column + radius + 1)
    if row_start >= row_end or column_start >= column_end:
        return 0.0
    rows, columns = np.ogrid[row_start:row_end, column_start:column_end]
    circular_region = (rows - row) ** 2 + (columns - column) ** 2 <= radius**2
    if not circular_region.any():
        return 0.0
    local_hsv = hsv_image[row_start:row_end, column_start:column_end]
    hue, saturation, value = (
        local_hsv[..., 0],
        local_hsv[..., 1],
        local_hsv[..., 2],
    )
    # OpenCV hue is 0--179. The PIVC hub is blue; use both hue and saturation
    # so grey/white catheter and dressing material cannot become hub evidence.
    blue = (
        (hue >= 90)
        & (hue <= 140)
        & (saturation >= 60)
        & (value >= 35)
        & circular_region
    )
    strength = (saturation.astype(np.float64) / 255.0) * (
        value.astype(np.float64) / 255.0
    )
    return float(strength[blue].sum() / circular_region.sum())


def orient_path_with_hub_evidence(
    path: np.ndarray, image_rgb: np.ndarray, pivc_width: float
) -> tuple[np.ndarray, dict]:
    """Orient a completed centreline from hub to insertion using image evidence.

    The chain builder deliberately does not rely on image rows, so its first
    point has no clinical direction. Here, the local saturated-blue hub
    signature is evaluated around *both* terminal points. If one side is not
    decisively more hub-like, the safe response is to reject orientation rather
    than silently assign top/left/right as the hub.
    """

    centreline = np.asarray(path, dtype=np.float64)
    if centreline.ndim != 2 or centreline.shape[1] != 2 or len(centreline) < 2:
        raise RuntimeError("Hub-side endpoint is ambiguous: PIVC path is invalid.")
    if not math.isfinite(float(pivc_width)) or pivc_width <= 0:
        raise RuntimeError("Hub-side endpoint is ambiguous: PIVC width is invalid.")
    rgb_image = _rgb_uint8(image_rgb)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    radius = int(max(12, round(3.0 * float(pivc_width))))
    radius = min(radius, max(12, min(hsv_image.shape[:2]) // 2))
    scores = [
        _endpoint_blue_hub_score(hsv_image, centreline[index], radius)
        for index in (0, -1)
    ]
    strongest = max(scores)
    weakest = min(scores)
    # The absolute gate prevents a random faint blue pixel from selecting a
    # direction. The relative gate rejects two similarly blue endpoints.
    if strongest < 0.005 or strongest - weakest < max(0.003, 0.50 * strongest):
        raise RuntimeError(
            "Hub-side endpoint is ambiguous: image evidence does not "
            "distinguish the PIVC hub from the insertion side."
        )
    hub_endpoint_index = int(np.argmax(scores))
    oriented = centreline.copy()
    if hub_endpoint_index == 1:
        oriented = oriented[::-1].copy()
    return oriented, {
        "hub_endpoint_index": hub_endpoint_index,
        "blue_hub_scores": [float(score) for score in scores],
        "sample_radius_px": radius,
    }


def _processing_result(case: ValidationCase) -> ValidationResult:
    return rejected_result(case, "processing", "Measurement has not completed.")


def process_segmentation(
    case: ValidationCase, output: SegmentationOutput, config: ValidationConfig
) -> tuple[ValidationResult, dict]:
    """Run conservative reconstruction, mark repair, and calibration.

    Every expected geometry failure becomes a result with an explicit stage,
    allowing the later batch runner to retain a rejected image rather than
    losing the whole validation run.
    """
    output = resize_segmentation_output(
        output,
        maximum_dimension=config.processing_max_dimension,
    )
    result = _processing_result(case)
    result.pivc_detected = bool(np.asarray(output.pivc_mask).any())
    result.pivc_confidence = float(output.pivc_confidence)
    mark_centres = _mark_centres(output.mark_masks)
    result.marks_detected = len(mark_centres)
    diagnostics: dict = {
        "image_rgb": np.asarray(output.image_rgb),
        "pivc_mask": np.asarray(output.pivc_mask).astype(bool),
        "mark_masks": [np.asarray(mask).astype(bool) for mask in output.mark_masks],
        "mark_centres": mark_centres,
        "components": [],
    }
    if not result.pivc_detected:
        result.rejection_stage = "segmentation"
        result.rejection_reason = "No usable PIVC mask was detected."
        return result, diagnostics

    components = component_paths_from_mask(output.pivc_mask)
    diagnostics["components"] = components
    if not components:
        result.rejection_stage = "components"
        result.rejection_reason = "No significant PIVC component path was extracted."
        return result, diagnostics

    try:
        reconstruction = reconstruct_continuous_centerline(components)
    except RuntimeError as exc:
        result.rejection_stage = "reconstruction"
        result.rejection_reason = str(exc)
        return result, diagnostics
    diagnostics["reconstruction"] = reconstruction

    retained_ids = set(reconstruction["component_ids"])
    retained_widths = [
        float(component["median_width"])
        for component in components
        if int(component["component_id"]) in retained_ids
    ]
    pivc_width = float(np.median(retained_widths)) if retained_widths else 0.0
    diagnostics["pivc_width"] = pivc_width

    try:
        repaired = repair_centerline_through_marks(
            reconstruction["path"], mark_centres, pivc_width
        )
    except RuntimeError as exc:
        result.rejection_stage = "mark_repair"
        result.rejection_reason = str(exc)
        return result, diagnostics
    diagnostics["repair"] = repaired
    # Mark count is a calibration prerequisite, independent of endpoint
    # direction. Preserve this explicit failure stage instead of masking it
    # with a later image-evidence orientation rejection.
    if len(mark_centres) < 2:
        result.rejection_stage = "calibration"
        result.rejection_reason = "Calibration requires at least two detected marks."
        return result, diagnostics
    try:
        corrected_path, endpoint_evidence = orient_path_with_hub_evidence(
            repaired["path"], output.image_rgb, pivc_width
        )
    except RuntimeError as exc:
        result.rejection_stage = "endpoint_orientation"
        result.rejection_reason = str(exc)
        return result, diagnostics
    diagnostics["endpoint_evidence"] = endpoint_evidence
    diagnostics["corrected_path"] = corrected_path
    diagnostics["repaired_mark_indices"] = repaired["repaired_mark_indices"]
    diagnostics["endpoints"] = {
        "hub": corrected_path[0],
        "insertion": corrected_path[-1],
    }
    result.corrected_centreline_px = float(
        np.linalg.norm(np.diff(corrected_path, axis=0), axis=1).sum()
    )
    result.endpoints_review = "Unreviewed"

    try:
        calibration = calibrate_length_from_marks(
            corrected_path,
            mark_centres,
            pivc_width,
            known_spacing_cm=config.known_mark_spacing_cm,
        )
    except RuntimeError as exc:
        result.rejection_stage = "calibration"
        result.rejection_reason = str(exc)
        return result, diagnostics
    diagnostics["calibration"] = calibration
    diagnostics["projected_mark_points"] = calibration["projected_mark_points"]

    result.consecutive_mark_spacings_px = [
        float(value) for value in calibration["spacing_pixels"]
    ]
    result.pixels_per_cm = float(calibration["pixels_per_cm"])
    result.estimated_length_cm = float(calibration["length_cm"])
    result.status, result.absolute_error_cm = classify_accuracy(
        result.estimated_length_cm,
        case.known_length_cm,
        config.accuracy_tolerance_cm,
    )
    result.rejection_stage = ""
    result.rejection_reason = ""
    return result, diagnostics


def _overlay_base_image(case: ValidationCase, diagnostics: dict) -> np.ndarray:
    """Return an RGB image for an overlay, including partial failures."""

    image_rgb = diagnostics.get("image_rgb")
    if image_rgb is not None:
        image = np.asarray(image_rgb)
        if image.ndim == 2:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] >= 3:
            return image[..., :3].astype(np.uint8).copy()

    loaded_bgr = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
    if loaded_bgr is not None:
        return cv2.cvtColor(loaded_bgr, cv2.COLOR_BGR2RGB)
    return np.full((720, 960, 3), 32, dtype=np.uint8)


def _overlay_point(point: np.ndarray | Sequence[float], scale: float) -> tuple[int, int]:
    """Convert a stored ``(row, column)`` point to a displayed ``(x, y)``."""

    row, column = np.asarray(point, dtype=np.float64)[:2]
    return int(round(column * scale)), int(round(row * scale))


def _draw_overlay_path(
    image: np.ndarray,
    path: np.ndarray | Sequence[Sequence[float]] | None,
    scale: float,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    if path is None:
        return
    points = np.asarray(path, dtype=np.float64)
    if len(points) < 2:
        return
    displayed = np.asarray(
        [_overlay_point(point, scale) for point in points], dtype=np.int32
    ).reshape((-1, 1, 2))
    cv2.polylines(image, [displayed], False, colour, thickness, cv2.LINE_AA)


def _overlay_lines(text: str, max_width: int, font_scale: float) -> list[str]:
    """Wrap footer text without placing metadata over the image."""

    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        if width <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def render_validation_overlay(
    case: ValidationCase,
    result: ValidationResult,
    diagnostics: dict,
    output_path: Path,
) -> Path:
    """Render one non-destructive, orientation-independent diagnostic image.

    Geometry is retained in the original image coordinate system.  Only the
    rendered display is enlarged when needed for inspection; an information
    footer keeps labels away from the PIVC itself.
    """

    source_rgb = _overlay_base_image(case, diagnostics)
    source_height, source_width = source_rgb.shape[:2]
    display_scale = min(
        1.0,
        1200.0 / float(max(source_height, source_width)),
    )
    display_width = int(round(source_width * display_scale))
    display_height = int(round(source_height * display_scale))
    image = cv2.resize(
        source_rgb, (display_width, display_height), interpolation=cv2.INTER_LINEAR
    )

    mask = diagnostics.get("pivc_mask")
    if mask is not None:
        mask_display = cv2.resize(
            np.asarray(mask, dtype=np.uint8),
            (display_width, display_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        mask_layer = image.copy()
        mask_layer[mask_display] = (70, 220, 90)
        image = cv2.addWeighted(image, 0.58, mask_layer, 0.42, 0.0)

    line_thickness = max(2, int(round(2.0 * display_scale)))
    marker_radius = max(5, int(round(5.0 * display_scale)))
    annotation_font_scale = min(
        1.2, max(0.5, 0.56 * math.sqrt(display_scale))
    )
    text_thickness = max(1, min(3, int(round(display_scale))))

    reconstruction = diagnostics.get("reconstruction", {})
    for bridge in reconstruction.get("bridges", []):
        _draw_overlay_path(image, bridge, display_scale, (255, 0, 255), line_thickness + 1)

    repair = diagnostics.get("repair", {})
    for repair_record in repair.get("repairs", []):
        _draw_overlay_path(
            image,
            repair_record.get("replacement"),
            display_scale,
            (0, 220, 255),
            line_thickness + 1,
        )

    corrected_path = diagnostics.get("corrected_path")
    _draw_overlay_path(image, corrected_path, display_scale, (255, 235, 0), line_thickness)

    for centre in diagnostics.get("mark_centres", []):
        point = _overlay_point(centre, display_scale)
        cv2.drawMarker(
            image, point, (255, 100, 0), cv2.MARKER_TILTED_CROSS,
            marker_radius * 2, line_thickness, cv2.LINE_AA,
        )

    projected_points = np.asarray(
        diagnostics.get("projected_mark_points", []), dtype=np.float64
    )
    if projected_points.ndim == 2 and len(projected_points) >= 1:
        for point in projected_points:
            cv2.circle(
                image, _overlay_point(point, display_scale), marker_radius,
                (255, 0, 180), -1, cv2.LINE_AA,
            )

        spacing_values = diagnostics.get("calibration", {}).get(
            "spacing_pixels", result.consecutive_mark_spacings_px
        )
        for index, spacing in enumerate(spacing_values):
            if index + 1 >= len(projected_points):
                break
            first = _overlay_point(projected_points[index], display_scale)
            second = _overlay_point(projected_points[index + 1], display_scale)
            cv2.line(image, first, second, (255, 145, 0), line_thickness + 2, cv2.LINE_AA)
            midpoint = (np.asarray(first, dtype=np.float64) + np.asarray(second, dtype=np.float64)) / 2.0
            direction = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction))
            label = f"{float(spacing):.2f} px = 1.00 cm"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, annotation_font_scale, text_thickness
            )[0]
            label_origin = midpoint.copy()
            if direction_norm > 0:
                normal = np.asarray([-direction[1], direction[0]]) / direction_norm
                offset = label_size[0] / 2.0 + max(18.0, 10.0 * display_scale)
                candidates = [
                    midpoint + normal * offset,
                    midpoint - normal * offset,
                ]
                for candidate in candidates:
                    origin = candidate - np.asarray([label_size[0] / 2.0, -label_size[1] / 2.0])
                    if (
                        4 <= origin[0]
                        and origin[0] + label_size[0] <= display_width - 4
                        and label_size[1] + 4 <= origin[1] <= display_height - 4
                    ):
                        label_origin = origin
                        break
            label_origin = tuple(np.asarray(label_origin, dtype=np.int32))
            cv2.putText(
                image, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX,
                annotation_font_scale, (255, 255, 255),
                text_thickness, cv2.LINE_AA,
            )

    endpoints = diagnostics.get("endpoints", {})
    hub = endpoints.get("hub") if isinstance(endpoints, dict) else None
    insertion = endpoints.get("insertion") if isinstance(endpoints, dict) else None
    if hub is None and corrected_path is not None and len(corrected_path) >= 2:
        hub, insertion = corrected_path[0], corrected_path[-1]
    for point, colour, label in (
        (hub, (255, 220, 0), "hub"),
        (insertion, (0, 210, 255), "insertion"),
    ):
        if point is None:
            continue
        position = _overlay_point(point, display_scale)
        cv2.circle(image, position, marker_radius + 2, colour, -1, cv2.LINE_AA)
        cv2.putText(
            image, label, (position[0] + marker_radius, position[1] - marker_radius),
            cv2.FONT_HERSHEY_SIMPLEX, annotation_font_scale, colour,
            text_thickness, cv2.LINE_AA,
        )

    estimated = "N/A" if result.estimated_length_cm is None else f"{result.estimated_length_cm:.3f} cm"
    error = "N/A" if result.absolute_error_cm is None else f"{result.absolute_error_cm:.3f} cm"
    footer_text = [
        f"{case.image_path.name} | {case.dressing_condition} | known: {case.known_length_cm:.3f} cm",
        f"estimated: {estimated} | absolute error: {error} | status: {result.status}",
    ]
    if result.status == "REJECTED":
        footer_text.append(
            f"rejected at {result.rejection_stage or 'processing'}: {result.rejection_reason or 'No reason recorded.'}"
        )

    font_scale = annotation_font_scale
    wrapped = [line for text in footer_text for line in _overlay_lines(text, display_width - 40, font_scale)]
    line_height = max(24, int(round(28 * display_scale)))
    footer_height = 28 + line_height * len(wrapped)
    rendered = np.full((display_height + footer_height, display_width, 3), 24, dtype=np.uint8)
    rendered[:display_height] = image
    for index, line in enumerate(wrapped):
        cv2.putText(
            rendered, str(line), (20, display_height + 22 + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245),
            text_thickness, cv2.LINE_AA,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, 88]
        if output_path.suffix.lower() in {".jpg", ".jpeg"}
        else []
    )
    if not cv2.imwrite(str(output_path), output_bgr, parameters):
        raise RuntimeError(f"Could not save diagnostic overlay: {output_path}")
    result.diagnostic_path = str(output_path)
    result.diagnostic_available = True
    return output_path


def _save_case_diagnostic(
    case: ValidationCase,
    result: ValidationResult,
    diagnostics: dict,
    diagnostics_dir: Path | None,
) -> None:
    """Attempt diagnostic persistence without changing measurement semantics."""

    if diagnostics_dir is None:
        return

    output_path = Path(diagnostics_dir) / f"{result.case_id.replace('/', '__')}.jpg"
    render_diagnostics = dict(diagnostics)
    if render_diagnostics.get("image_rgb") is None:
        source_bgr = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            result.diagnostic_path = str(output_path)
            result.diagnostic_available = False
            result.diagnostic_error = "Source image is unreadable for diagnostic overlay."
            return
        render_diagnostics["image_rgb"] = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)

    try:
        render_validation_overlay(case, result, render_diagnostics, output_path)
        result.diagnostic_error = ""
    except Exception as exc:
        result.diagnostic_path = str(output_path)
        result.diagnostic_available = False
        result.diagnostic_error = f"Could not render diagnostic overlay: {exc}"


def process_validation_case(
    case: ValidationCase,
    model,
    config: ValidationConfig,
    diagnostics_dir: Path | None = None,
    *,
    predictor: Callable[[object, Path, ValidationConfig], SegmentationOutput] = predict_segmentation,
) -> ValidationResult:
    """Process one image and convert all prediction/geometry failures to data."""

    try:
        output = predictor(model, case.image_path, config)
    except SegmentationPredictionError as exc:
        result = rejected_result(case, "segmentation", str(exc))
        _save_case_diagnostic(case, result, {}, diagnostics_dir)
        return result
    except Exception as exc:
        result = rejected_result(
            case, "segmentation", f"Unexpected prediction error: {exc}"
        )
        _save_case_diagnostic(case, result, {}, diagnostics_dir)
        return result
    try:
        result, diagnostics = process_segmentation(case, output, config)
        _save_case_diagnostic(case, result, diagnostics, diagnostics_dir)
        return result
    except Exception as exc:  # final guard: a single case never terminates a batch
        return rejected_result(case, "processing", f"Unexpected processing error: {exc}")


_CHECKPOINT_SCHEMA_VERSION = 1
_RESULT_STATUSES = {"ACCURATE", "NOT ACCURATE", "REJECTED"}


def _case_id(case: ValidationCase) -> str:
    return "/".join((case.length_group, case.dressing_condition, case.image_path.name))


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_path_from(model, model_path: Path | None) -> Path:
    if model_path is not None:
        return Path(model_path)
    for attribute in ("ckpt_path", "model_path", "weights"):
        candidate = getattr(model, attribute, None)
        if candidate:
            if isinstance(candidate, (str, os.PathLike)):
                return Path(candidate)
            try:
                return Path(candidate)
            except TypeError:
                continue
    return Path("<in-memory-model>")


def _model_metadata(model_path: Path) -> dict[str, str | None]:
    path = Path(model_path)
    try:
        display_path = str(path.resolve())
    except OSError:
        display_path = str(path)
    return {"path": display_path, "sha256": _sha256_file(path)}


def configuration_fingerprint(config: ValidationConfig, model_path: Path) -> str:
    """Return a stable fingerprint for model bytes, model path, and run settings."""

    if not isinstance(config, ValidationConfig):
        raise TypeError("config must be a ValidationConfig")
    payload = {
        "configuration": asdict(config),
        "model": _model_metadata(Path(model_path)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_fingerprint(cases: Sequence[ValidationCase]) -> str:
    """Hash case identities and source bytes so changed inputs invalidate resume."""

    digest = hashlib.sha256()
    for case in sorted(cases, key=_case_id):
        path = Path(case.image_path)
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        digest.update(_case_id(case).encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.encode("utf-8"))
        digest.update(b"\0")
        file_hash = _sha256_file(path)
        digest.update((file_hash or "<missing>").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": os.sys.version.split()[0]}
    for package in ("ultralytics", "opencv-python", "numpy", "scipy", "scikit-image"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _json_safe(value):
    """Convert checkpoint values into strict JSON-compatible primitives."""

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Checkpoint value is not JSON serializable: {type(value).__name__}")


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    """Persist a checkpoint using a sibling temporary file and atomic replace."""

    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(payload)
    checkpoint.setdefault("schema_version", _CHECKPOINT_SCHEMA_VERSION)
    checkpoint.setdefault(
        "updated_at_utc", datetime.now(timezone.utc).isoformat()
    )
    checkpoint.setdefault("package_versions", _package_versions())
    checkpoint.setdefault("results", [])
    safe_checkpoint = _json_safe(checkpoint)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(safe_checkpoint, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint, returning an empty payload when no run exists yet."""

    path = Path(path)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint must contain a JSON object: {path}")
    return payload


def _result_from_record(record: dict) -> ValidationResult:
    if not isinstance(record, dict):
        raise ValueError("checkpoint result must be an object")
    known_fields = {item.name for item in fields(ValidationResult)}
    values = {key: value for key, value in record.items() if key in known_fields}
    missing = {"case_id", "filename", "image_path", "length_group", "dressing_condition", "known_length_cm"} - set(values)
    if missing:
        raise ValueError(f"checkpoint result is missing fields: {sorted(missing)}")
    return ValidationResult(**values)


def _diagnostic_metadata_complete(record: ValidationResult) -> bool:
    if record.diagnostic_available:
        return bool(record.diagnostic_path) and Path(record.diagnostic_path).is_file()
    # A diagnostic failure is itself retained metadata. This allows unreadable
    # images or disk errors to remain completed attempts without retry loops.
    return bool(record.diagnostic_error)


def _matching_complete_result(
    record: ValidationResult, case: ValidationCase, fingerprint: str
) -> bool:
    return (
        record.case_id == _case_id(case)
        and record.status in _RESULT_STATUSES
        and record.config_fingerprint == fingerprint
        and _diagnostic_metadata_complete(record)
    )


def run_validation_batch(
    cases: Sequence[ValidationCase],
    model,
    config: ValidationConfig,
    output_dir: Path,
    processor: Callable = process_validation_case,
    *,
    model_path: Path | None = None,
) -> list[ValidationResult]:
    """Process cases sequentially with atomic per-case checkpointing and resume."""

    if not isinstance(config, ValidationConfig):
        raise TypeError("config must be a ValidationConfig")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    ordered_cases = sorted(cases, key=_case_id)
    case_ids = [_case_id(case) for case in ordered_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Validation cases contain duplicate case_id values")
    cases_by_id = dict(zip(case_ids, ordered_cases))
    resolved_model_path = _model_path_from(model, model_path)
    fingerprint = configuration_fingerprint(config, resolved_model_path)
    inputs_fingerprint = _input_fingerprint(ordered_cases)
    checkpoint_path = output_dir / "checkpoint.json"
    existing = load_checkpoint(checkpoint_path)
    existing_matches = (
        existing.get("config_fingerprint") == fingerprint
        and existing.get("input_fingerprint") == inputs_fingerprint
    )
    prior_results: dict[str, ValidationResult] = {}
    if existing_matches:
        for record in existing.get("results", []):
            try:
                result = _result_from_record(record)
            except (TypeError, ValueError):
                continue
            prior_results[result.case_id] = result

    # Carry every already-complete matching record into the new in-memory
    # state before retrying anything. Otherwise saving an early retry could
    # accidentally discard a later case that is skipped from the checkpoint.
    results_by_case: dict[str, ValidationResult] = {
        case_id: prior_results[case_id]
        for case_id in case_ids
        if case_id in prior_results
        and _matching_complete_result(
            prior_results[case_id], cases_by_id[case_id], fingerprint
        )
    }
    for index, case in enumerate(ordered_cases, start=1):
        stable_id = _case_id(case)
        prior = prior_results.get(stable_id)
        if prior is not None and _matching_complete_result(prior, case, fingerprint):
            results_by_case[stable_id] = prior
            print(f"[{index}/{len(ordered_cases)}] {stable_id}: SKIPPED (checkpoint)")
            continue

        try:
            result = processor(case, model, config, diagnostics_dir)
            if isinstance(result, dict):
                result = _result_from_record(result)
            if not isinstance(result, ValidationResult):
                raise TypeError("processor did not return ValidationResult")
        except Exception as exc:  # isolate one image; KeyboardInterrupt is not caught
            result = rejected_result(
                case,
                "unexpected_error",
                f"Unexpected validation error: {exc}",
            )
        result.config_fingerprint = fingerprint
        results_by_case[stable_id] = result
        checkpoint_payload = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_versions": _package_versions(),
            "config": asdict(config),
            "config_fingerprint": fingerprint,
            "model": _model_metadata(resolved_model_path),
            "input_fingerprint": inputs_fingerprint,
            "results": [
                asdict(results_by_case[case_id])
                for case_id in case_ids
                if case_id in results_by_case
            ],
        }
        save_checkpoint_atomic(checkpoint_path, checkpoint_payload)
        print(f"[{index}/{len(ordered_cases)}] {stable_id}: {result.status}")

    ordered_results = [results_by_case[case_id] for case_id in case_ids if case_id in results_by_case]
    counts = Counter(result.status for result in ordered_results)
    print(
        "Batch summary: "
        f"{len(ordered_results)}/{len(ordered_cases)} complete | "
        f"ACCURATE={counts.get('ACCURATE', 0)} | "
        f"NOT ACCURATE={counts.get('NOT ACCURATE', 0)} | "
        f"REJECTED={counts.get('REJECTED', 0)}"
    )
    return ordered_results


__all__ = [
    "DRESSING_GROUPS",
    "IMAGE_SUFFIXES",
    "LENGTHS_CM",
    "ValidationCase",
    "ValidationConfig",
    "ValidationResult",
    "SegmentationOutput",
    "SegmentationPredictionError",
    "classify_accuracy",
    "component_paths_from_mask",
    "configuration_fingerprint",
    "discover_validation_cases",
    "load_checkpoint",
    "orient_path_with_hub_evidence",
    "predict_segmentation",
    "process_segmentation",
    "process_validation_case",
    "render_validation_overlay",
    "rejected_result",
    "run_validation_batch",
    "save_checkpoint_atomic",
    "validate_case_balance",
]
