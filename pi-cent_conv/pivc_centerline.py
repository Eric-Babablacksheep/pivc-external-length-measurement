"""Conservative reconstruction of a continuous PIVC centreline."""

from __future__ import annotations

import math

import numpy as np


def _unit(vector):
    length = float(np.linalg.norm(vector))
    return None if length == 0 else vector / length


def _angle_degrees(first, second):
    if first is None or second is None:
        return float("inf")
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _principal_axis(components):
    points = np.concatenate(
        [np.asarray(item["path_array"], dtype=np.float64) for item in components]
    )
    centred = points - points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(centred, full_matrices=False)
    axis = right_vectors[0]
    # Images in this project are acquired hub-up. Positive projection therefore
    # runs from the hub side toward the insertion side (increasing image row).
    if axis[0] < 0:
        axis = -axis
    return axis


def _orient_path(path, axis):
    path = np.asarray(path, dtype=np.float64)
    if float(np.dot(path[-1] - path[0], axis)) < 0:
        return path[::-1].copy()
    return path.copy()


def _densify_polyline(path, maximum_step=1.0):
    dense_parts = []
    for start, end in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(end - start))
        sample_count = max(2, int(math.ceil(distance / maximum_step)) + 1)
        dense_parts.append(np.linspace(start, end, sample_count)[:-1])
    dense_parts.append(path[-1:])
    return np.concatenate(dense_parts)


def _smooth_bridge(start, end, start_tangent, end_tangent):
    gap = float(np.linalg.norm(end - start))
    sample_count = max(3, int(math.ceil(gap)) + 1)
    t = np.linspace(0.0, 1.0, sample_count)[:, None]
    tangent_scale = 0.35 * gap
    first_control = start_tangent * tangent_scale
    second_control = end_tangent * tangent_scale
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * start + h10 * first_control + h01 * end + h11 * second_control


def _robust_travel_tangent(path, at_start, width):
    path = np.asarray(path, dtype=np.float64)
    if not at_start:
        path = path[::-1]
    target_distance = max(60.0, 8.0 * float(width))
    travelled = 0.0
    end_index = 1
    while end_index < len(path) and travelled < target_distance:
        travelled += float(np.linalg.norm(path[end_index] - path[end_index - 1]))
        end_index += 1
    window = path[:end_index]
    if len(window) < 2:
        return None
    centred = window - window.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(centred, full_matrices=False)
    tangent = right_vectors[0]
    chord = window[-1] - window[0]
    if float(np.dot(tangent, chord)) < 0:
        tangent = -tangent
    if not at_start:
        tangent = -tangent
    return _unit(tangent)


def _connection_metrics(first_path, first_item, second_path, second_item):
    first_tangent = _robust_travel_tangent(
        first_path, at_start=False, width=first_item["median_width"]
    )
    second_tangent = _robust_travel_tangent(
        second_path, at_start=True, width=second_item["median_width"]
    )
    bridge_vector = second_path[0] - first_path[-1]
    bridge_direction = _unit(bridge_vector)
    gap = float(np.linalg.norm(bridge_vector))
    widths = [
        float(first_item["median_width"]),
        float(second_item["median_width"]),
    ]
    average_width = float(np.mean(widths))
    return {
        "from_component": int(first_item["component_id"]),
        "to_component": int(second_item["component_id"]),
        "gap_px": gap,
        "normalized_gap": gap / average_width,
        "angle_1": _angle_degrees(first_tangent, bridge_direction),
        "angle_2": _angle_degrees(second_tangent, bridge_direction),
        "width_ratio": max(widths) / min(widths),
        "first_tangent": first_tangent,
        "second_tangent": second_tangent,
    }


def reconstruct_continuous_centerline(
    component_paths,
    *,
    minimum_width_fraction=0.50,
    maximum_normalized_gap=2.50,
    maximum_connection_angle=55.0,
    maximum_width_ratio=2.0,
):
    """Return one ordered hub-to-insertion path or reject an unsafe bridge.

    Narrow segmentation fragments are excluded relative to the typical PIVC
    width. Remaining component paths are ordered along their shared principal
    axis and joined with tangent-preserving cubic Hermite bridges.
    """
    if not component_paths:
        raise RuntimeError("No PIVC components are available.")

    widths = np.asarray(
        [float(item["median_width"]) for item in component_paths],
        dtype=np.float64,
    )
    reference_width = float(np.median(widths))
    retained = [
        item
        for item in component_paths
        if float(item["median_width"])
        >= minimum_width_fraction * reference_width
    ]
    if not retained:
        raise RuntimeError("No PIVC components passed the width gate.")

    path_options = {
        (index, orientation): (
            np.asarray(item["path_array"], dtype=np.float64)
            if orientation == 0
            else np.asarray(item["path_array"], dtype=np.float64)[::-1].copy()
        )
        for index, item in enumerate(retained)
        for orientation in (0, 1)
    }
    all_mask = (1 << len(retained)) - 1
    states = {}
    for index in range(len(retained)):
        for orientation in (0, 1):
            states[(1 << index, index, orientation)] = (0.0, [(index, orientation)], [])

    for used_count in range(1, len(retained)):
        next_states = dict(states)
        for (mask, last_index, last_orientation), (cost, sequence, diagnostics) in states.items():
            if mask.bit_count() != used_count:
                continue
            first_item = retained[last_index]
            first_path = path_options[(last_index, last_orientation)]
            for next_index, second_item in enumerate(retained):
                if mask & (1 << next_index):
                    continue
                for next_orientation in (0, 1):
                    second_path = path_options[(next_index, next_orientation)]
                    metric = _connection_metrics(
                        first_path, first_item, second_path, second_item
                    )
                    unsafe = (
                        metric["normalized_gap"] > maximum_normalized_gap
                        or metric["angle_1"] > maximum_connection_angle
                        or metric["angle_2"] > maximum_connection_angle
                        or metric["width_ratio"] > maximum_width_ratio
                    )
                    if unsafe:
                        continue
                    transition_cost = (
                        metric["normalized_gap"]
                        + (metric["angle_1"] + metric["angle_2"]) / 90.0
                        + abs(math.log(metric["width_ratio"]))
                    )
                    key = (
                        mask | (1 << next_index), next_index, next_orientation
                    )
                    candidate = (
                        cost + transition_cost,
                        sequence + [(next_index, next_orientation)],
                        diagnostics + [metric],
                    )
                    if key not in next_states or candidate[0] < next_states[key][0]:
                        next_states[key] = candidate
        states = next_states

    complete = [
        value for (mask, _, _), value in states.items() if mask == all_mask
    ]
    if not complete:
        raise RuntimeError(
            "Reconstruction rejected: no rotation-independent chain can join "
            "all retained PIVC components without an unsafe gap."
        )
    _, best_sequence, best_diagnostics = min(complete, key=lambda value: value[0])
    ordered_items = [retained[index] for index, _ in best_sequence]
    ordered_paths = [
        _densify_polyline(path_options[(index, orientation)])
        for index, orientation in best_sequence
    ]
    bridges = []
    joined_parts = [ordered_paths[0]]
    connection_diagnostics = []

    for index in range(len(ordered_paths) - 1):
        first_path = ordered_paths[index]
        second_path = ordered_paths[index + 1]
        first_item = ordered_items[index]
        second_item = ordered_items[index + 1]
        diagnostic = best_diagnostics[index]
        connection_diagnostics.append(diagnostic)
        bridge = _smooth_bridge(
            first_path[-1],
            second_path[0],
            diagnostic["first_tangent"],
            diagnostic["second_tangent"],
        )
        bridges.append(bridge)
        joined_parts.extend([bridge[1:-1], second_path])

    continuous_path = np.concatenate(
        [part for part in joined_parts if len(part) > 0]
    )
    total_length = float(
        np.linalg.norm(np.diff(continuous_path, axis=0), axis=1).sum()
    )
    return {
        "path": continuous_path,
        "real_paths": ordered_paths,
        "bridges": bridges,
        "component_ids": [int(item["component_id"]) for item in ordered_items],
        "excluded_component_ids": [
            int(item["component_id"])
            for item in component_paths
            if item not in retained
        ],
        "connections": connection_diagnostics,
        "total_length_px": total_length,
        "hub_endpoint": continuous_path[0],
        "insertion_endpoint": continuous_path[-1],
        "is_continuous": True,
    }


def _path_length(path):
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _anchor_index(path, centre_index, travel_distance, direction):
    index = centre_index
    travelled = 0.0
    while 0 <= index + direction < len(path):
        next_index = index + direction
        travelled += float(np.linalg.norm(path[next_index] - path[index]))
        index = next_index
        if travelled >= travel_distance:
            break
    return index


def repair_centerline_through_marks(
    path,
    mark_centres,
    pivc_width,
    *,
    maximum_mark_distance_factor=2.0,
    anchor_distance_factor=1.75,
):
    """Repair short mark-associated detours in an ordered PIVC centreline.

    The function changes only a local window around a detected mark that is
    close to the current centreline. The replacement passes through the mark
    centroid and follows the centreline directions on both sides.
    """
    repaired_path = np.asarray(path, dtype=np.float64).copy()
    if len(repaired_path) < 3 or pivc_width <= 0:
        raise RuntimeError("PIVC path or width is invalid for mark repair.")

    repair_records = []
    repaired_indices = []
    maximum_mark_distance = maximum_mark_distance_factor * float(pivc_width)
    anchor_distance = anchor_distance_factor * float(pivc_width)

    for mark_index, mark_centre in enumerate(mark_centres):
        mark = np.asarray(mark_centre, dtype=np.float64)
        distances = np.linalg.norm(repaired_path - mark, axis=1)
        if float(distances.min()) > maximum_mark_distance:
            continue

        nearby = np.flatnonzero(distances <= maximum_mark_distance)
        centre_index = int(np.median(nearby))
        start_index = _anchor_index(
            repaired_path, centre_index, anchor_distance, -1
        )
        end_index = _anchor_index(
            repaired_path, centre_index, anchor_distance, 1
        )
        if start_index >= centre_index or end_index <= centre_index:
            continue

        tangent_sample = min(12, centre_index - start_index, end_index - centre_index)
        start_tangent = _unit(
            repaired_path[start_index + tangent_sample]
            - repaired_path[start_index]
        )
        end_tangent = _unit(
            repaired_path[end_index]
            - repaired_path[end_index - tangent_sample]
        )
        through_tangent = _unit(
            repaired_path[end_index] - repaired_path[start_index]
        )
        if start_tangent is None or end_tangent is None or through_tangent is None:
            continue

        first_half = _smooth_bridge(
            repaired_path[start_index], mark, start_tangent, through_tangent
        )
        second_half = _smooth_bridge(
            mark, repaired_path[end_index], through_tangent, end_tangent
        )
        replacement = np.concatenate([first_half, second_half[1:]])
        original_section = repaired_path[start_index : end_index + 1].copy()
        repaired_path = np.concatenate(
            [
                repaired_path[:start_index],
                replacement,
                repaired_path[end_index + 1 :],
            ]
        )
        repaired_indices.append(mark_index)
        repair_records.append(
            {
                "mark_index": mark_index,
                "mark_centre": mark,
                "original_section": original_section,
                "replacement": replacement,
                "original_length_px": _path_length(original_section),
                "replacement_length_px": _path_length(replacement),
            }
        )

    return {
        "path": repaired_path,
        "repairs": repair_records,
        "repaired_mark_indices": repaired_indices,
        "total_length_px": _path_length(repaired_path),
    }


def _project_point_to_polyline(point, path):
    segment_starts = path[:-1]
    segment_vectors = path[1:] - path[:-1]
    segment_lengths_squared = np.einsum(
        "ij,ij->i", segment_vectors, segment_vectors
    )
    valid = segment_lengths_squared > 0
    parameters = np.zeros(len(segment_vectors), dtype=np.float64)
    parameters[valid] = np.einsum(
        "ij,ij->i",
        point - segment_starts[valid],
        segment_vectors[valid],
    ) / segment_lengths_squared[valid]
    parameters = np.clip(parameters, 0.0, 1.0)
    projections = segment_starts + parameters[:, None] * segment_vectors
    distances = np.linalg.norm(projections - point, axis=1)
    segment_index = int(np.argmin(distances))
    segment_lengths = np.sqrt(segment_lengths_squared)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    arc_position = (
        cumulative[segment_index]
        + parameters[segment_index] * segment_lengths[segment_index]
    )
    return {
        "point": projections[segment_index],
        "distance": float(distances[segment_index]),
        "arc_position": float(arc_position),
    }


def calibrate_length_from_marks(
    path,
    mark_centres,
    pivc_width,
    *,
    known_spacing_cm=1.0,
    maximum_projection_distance_factor=1.5,
    maximum_spacing_deviation=0.15,
):
    """Convert a corrected PIVC centreline from pixels to centimetres."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        raise RuntimeError("PIVC centreline is too short for calibration.")
    if len(mark_centres) < 2:
        raise RuntimeError("Calibration requires at least two detected marks.")
    if pivc_width <= 0 or known_spacing_cm <= 0:
        raise RuntimeError("Calibration width and known spacing must be positive.")

    projections = [
        _project_point_to_polyline(
            np.asarray(mark_centre, dtype=np.float64), path
        )
        for mark_centre in mark_centres
    ]
    maximum_distance = maximum_projection_distance_factor * float(pivc_width)
    distant = [
        index
        for index, projection in enumerate(projections)
        if projection["distance"] > maximum_distance
    ]
    if distant:
        raise RuntimeError(
            "Calibration rejected: mark(s) too far from the PIVC centreline: "
            f"{distant}."
        )

    projections.sort(key=lambda item: item["arc_position"])
    positions = np.asarray(
        [item["arc_position"] for item in projections], dtype=np.float64
    )
    spacing_pixels = np.diff(positions)
    if np.any(spacing_pixels <= 2.0 * float(pivc_width)):
        raise RuntimeError(
            "Calibration rejected: consecutive mark spacing is implausibly small."
        )

    median_spacing = float(np.median(spacing_pixels))
    relative_deviations = np.abs(spacing_pixels - median_spacing) / median_spacing
    if len(spacing_pixels) > 1 and float(relative_deviations.max()) > maximum_spacing_deviation:
        raise RuntimeError(
            "Calibration rejected: consecutive mark spacing is inconsistent "
            f"(maximum relative deviation={relative_deviations.max():.1%})."
        )

    pixels_per_cm = median_spacing / float(known_spacing_cm)
    total_length_px = _path_length(path)
    return {
        "spacing_pixels": spacing_pixels,
        "pixels_per_cm": pixels_per_cm,
        "length_pixels": total_length_px,
        "length_cm": total_length_px / pixels_per_cm,
        "projected_mark_points": np.asarray(
            [item["point"] for item in projections], dtype=np.float64
        ),
        "mark_projection_distances": np.asarray(
            [item["distance"] for item in projections], dtype=np.float64
        ),
        "spacing_relative_deviations": relative_deviations,
        "known_spacing_cm": float(known_spacing_cm),
    }
