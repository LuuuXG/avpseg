"""Split the AVP mask into left/right optic nerves, chiasm, and optic tracts."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects

from .morphometry import build_position_graph, skeleton_degrees, skeletonize_3d


LABELS = {
    1: "left_optic_nerve",
    2: "right_optic_nerve",
    3: "optic_chiasm",
    4: "left_optic_tract",
    5: "right_optic_tract",
}


@dataclass(frozen=True)
class EndpointCandidate:
    index: int
    coord: np.ndarray
    quadrant: str
    score: float


def clean_mask(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    mask = mask.astype(bool)
    if not np.any(mask):
        return mask
    return remove_small_objects(mask, min_size=min_component_voxels, connectivity=3)


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    component_img, n_components = ndi.label(mask)
    if n_components == 0:
        return mask.copy(), 0, []
    sizes = np.bincount(component_img.ravel())[1:]
    largest_label = int(np.argmax(sizes) + 1)
    return component_img == largest_label, int(n_components), [int(v) for v in sorted(sizes, reverse=True)]


def dijkstra(graph: list[list[tuple[int, float]]], source: int) -> tuple[np.ndarray, np.ndarray]:
    dist = np.full(len(graph), np.inf, dtype=float)
    prev = np.full(len(graph), -1, dtype=int)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist > dist[node]:
            continue
        for nb, weight in graph[node]:
            nd = cur_dist + weight
            if nd < dist[nb]:
                dist[nb] = nd
                prev[nb] = node
                heapq.heappush(heap, (nd, nb))
    return dist, prev


def path_to_source(prev: np.ndarray, endpoint: int) -> list[int]:
    path: list[int] = []
    node = int(endpoint)
    while node >= 0:
        path.append(node)
        node = int(prev[node])
    return path


def infer_chiasm_source(coords: np.ndarray, degrees: np.ndarray) -> int:
    branch_indices = np.flatnonzero(degrees >= 3)
    if branch_indices.size:
        center = coords[branch_indices].mean(axis=0)
    else:
        center = coords.mean(axis=0)
    return int(np.argmin(np.linalg.norm(coords - center, axis=1)))


def refine_chiasm_source_from_endpoints(
    coords: np.ndarray,
    graph: list[list[tuple[int, float]]],
    endpoints: dict[str, int],
) -> int:
    endpoint_distances = []
    for endpoint in endpoints.values():
        dist, _ = dijkstra(graph, int(endpoint))
        endpoint_distances.append(dist)

    distances = np.vstack(endpoint_distances)
    finite = np.all(np.isfinite(distances), axis=0)
    if not np.any(finite):
        raise ValueError("Could not refine chiasm source from selected endpoints.")

    endpoint_coords = np.vstack([coords[int(index)] for index in endpoints.values()])
    lower = np.percentile(endpoint_coords, 20, axis=0) - 3.0
    upper = np.percentile(endpoint_coords, 80, axis=0) + 3.0
    central = finite & np.all(coords >= lower, axis=1) & np.all(coords <= upper, axis=1)
    if not np.any(central):
        central = finite

    score = distances.sum(axis=0) + 0.25 * distances.max(axis=0)
    score[~central] = np.inf
    return int(np.argmin(score))


def classify_endpoint(coord: np.ndarray, center: np.ndarray) -> str:
    lr = "left" if coord[0] < center[0] else "right"
    ap = "nerve" if coord[1] > center[1] else "tract"
    return f"{lr}_{ap}"


def select_endpoints(
    coords: np.ndarray,
    endpoint_indices: np.ndarray,
    center: np.ndarray,
    dist: np.ndarray,
) -> dict[str, int]:
    candidates: list[EndpointCandidate] = []
    for index in endpoint_indices:
        coord = coords[int(index)]
        quadrant = classify_endpoint(coord, center)
        dx = abs(float(coord[0] - center[0]))
        dy = abs(float(coord[1] - center[1]))
        score = dist[int(index)] + 0.25 * dx + 0.25 * dy
        candidates.append(EndpointCandidate(int(index), coord, quadrant, float(score)))

    selected: dict[str, int] = {}
    for quadrant in ("left_nerve", "right_nerve", "left_tract", "right_tract"):
        matches = [c for c in candidates if c.quadrant == quadrant and np.isfinite(dist[c.index])]
        if matches:
            selected[quadrant] = max(matches, key=lambda c: c.score).index

    if len(selected) == 4:
        return selected

    remaining = sorted(
        [c for c in candidates if c.index not in set(selected.values()) and np.isfinite(dist[c.index])],
        key=lambda c: c.score,
        reverse=True,
    )
    for c in remaining:
        selected.setdefault(c.quadrant, c.index)
        if len(selected) == 4:
            break
    if len(selected) == 4:
        return selected

    used = set(selected.values())
    for quadrant in ("left_nerve", "right_nerve", "left_tract", "right_tract"):
        if quadrant in selected:
            continue
        fallback: list[EndpointCandidate] = []
        for index, coord in enumerate(coords):
            if index in used or classify_endpoint(coord, center) != quadrant or not np.isfinite(dist[index]):
                continue
            dx = abs(float(coord[0] - center[0]))
            dy = abs(float(coord[1] - center[1]))
            score = dist[index] + 0.25 * dx + 0.25 * dy
            fallback.append(EndpointCandidate(int(index), coord, quadrant, float(score)))
        if fallback:
            selected[quadrant] = max(fallback, key=lambda c: c.score).index
            used.add(selected[quadrant])
        if len(selected) == 4:
            break
    return selected


def cumulative_path_lengths(path: list[int], coords: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    lengths = np.zeros(len(path), dtype=float)
    for i in range(1, len(path)):
        step = (coords[path[i]] - coords[path[i - 1]]).astype(float) * spacing
        lengths[i] = lengths[i - 1] + float(np.linalg.norm(step))
    return lengths


def make_centerline_labels(
    coords: np.ndarray,
    source: int,
    prev: np.ndarray,
    endpoints: dict[str, int],
    spacing: np.ndarray,
    chiasm_path_mm: float,
) -> tuple[np.ndarray, dict[str, float]]:
    centerline_labels = np.zeros(len(coords), dtype=np.uint8)
    centerline_labels[source] = 3
    path_lengths: dict[str, float] = {}
    branch_to_label = {
        "left_nerve": 1,
        "right_nerve": 2,
        "left_tract": 4,
        "right_tract": 5,
    }
    for branch, endpoint in endpoints.items():
        path = path_to_source(prev, endpoint)
        path = list(reversed(path))
        lengths = cumulative_path_lengths(path, coords, spacing)
        path_lengths[f"{branch}_path_mm"] = float(lengths[-1]) if len(lengths) else 0.0
        label = branch_to_label[branch]
        for node, length in zip(path, lengths):
            if length <= chiasm_path_mm:
                centerline_labels[node] = 3
            elif centerline_labels[node] == 0:
                centerline_labels[node] = label
    return centerline_labels, path_lengths


def assign_volume_from_centerline(
    mask: np.ndarray,
    coords: np.ndarray,
    centerline_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labeled_skel = np.zeros(mask.shape, dtype=np.uint8)
    valid = centerline_labels > 0
    labeled_skel[tuple(coords[valid].T)] = centerline_labels[valid]
    if not np.any(valid):
        raise ValueError("No labeled centerline voxels were found.")

    _, nearest = ndi.distance_transform_edt(labeled_skel == 0, return_indices=True)
    out = np.zeros(mask.shape, dtype=np.uint8)
    nearest_labels = labeled_skel[nearest[0], nearest[1], nearest[2]]
    out[mask] = nearest_labels[mask]
    full_centerline = np.zeros(mask.shape, dtype=np.uint8)
    full_centerline[tuple(coords.T)] = nearest_labels[tuple(coords.T)]
    return out, full_centerline


def apply_compact_chiasm_core(
    labels: np.ndarray,
    centerline: np.ndarray,
    mask: np.ndarray,
    source_coord: np.ndarray,
    spacing: np.ndarray,
    radii_mm: np.ndarray,
) -> np.ndarray:
    pts = np.argwhere(mask)
    scaled = ((pts - source_coord.astype(float)) * spacing) / radii_mm
    core_pts = pts[np.sum(scaled * scaled, axis=1) <= 1.0]

    out = labels.copy()
    core_mask = np.zeros(mask.shape, dtype=bool)
    if core_pts.size:
        core_mask[tuple(core_pts.T)] = True
        out[core_mask] = 3

    branch_centerline = centerline.copy()
    branch_centerline[branch_centerline == 3] = 0
    if not np.any(branch_centerline > 0):
        return out

    non_core_chiasm = (out == 3) & ~core_mask
    if np.any(non_core_chiasm):
        _, nearest = ndi.distance_transform_edt(branch_centerline == 0, return_indices=True)
        nearest_branch = branch_centerline[nearest[0], nearest[1], nearest[2]]
        out[non_core_chiasm] = nearest_branch[non_core_chiasm]
    return out


def separate_nerve_tract_pairs(labels: np.ndarray, max_iterations: int) -> tuple[np.ndarray, int]:
    out = labels.copy()
    structure = np.ones((3, 3, 3), dtype=bool)
    converted = 0
    for _ in range(max_iterations):
        to_chiasm = np.zeros(out.shape, dtype=bool)
        for nerve_label, tract_label in ((1, 4), (2, 5)):
            nerve = out == nerve_label
            tract = out == tract_label
            nerve_contact = nerve & ndi.binary_dilation(tract, structure=structure)
            tract_contact = tract & ndi.binary_dilation(nerve, structure=structure)
            to_chiasm |= nerve_contact | tract_contact
        n_changed = int(np.count_nonzero(to_chiasm))
        if n_changed == 0:
            break
        out[to_chiasm] = 3
        converted += n_changed
    return out, converted


def repair_missing_pair_components(
    labels: np.ndarray,
    label_pairs: tuple[tuple[int, int], ...] = ((1, 2), (4, 5)),
    min_component_voxels: int = 10,
) -> tuple[np.ndarray, dict[str, int]]:
    """Recover a missing left/right label when both sides were assigned the same label.

    The centerline scaffold is built from the largest foreground component. If an optic
    nerve is disconnected from that scaffold, nearest-centerline assignment can label
    both separated nerve components as the same side. This post-pass splits those
    obvious duplicate components along image axis 0; pipeline-level world-coordinate
    enforcement can still swap left/right labels afterward if needed.
    """
    out = labels.copy()
    repaired: dict[str, int] = {}
    for left_label, right_label in label_pairs:
        left_count = int(np.count_nonzero(out == left_label))
        right_count = int(np.count_nonzero(out == right_label))
        if (left_count == 0) == (right_count == 0):
            continue

        present_label = left_label if left_count else right_label
        component_img, n_components = ndi.label(out == present_label)
        components: list[tuple[float, int, int]] = []
        for component in range(1, n_components + 1):
            pts = np.argwhere(component_img == component)
            size = int(pts.shape[0])
            if size >= min_component_voxels:
                components.append((float(np.mean(pts[:, 0])), component, size))

        if len(components) < 2:
            continue

        components.sort(key=lambda item: item[0])
        split_at = (components[0][0] + components[-1][0]) / 2.0
        changed = 0
        for centroid_x, component, size in components:
            target_label = left_label if centroid_x <= split_at else right_label
            out[component_img == component] = target_label
            changed += size
        repaired[f"{left_label}_{right_label}_component_voxels"] = int(changed)
    return out, repaired


def repair_nerve_tract_orientation(
    labels: np.ndarray,
    min_component_voxels: int = 1,
    margin_voxels: float = 2.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reassign disconnected same-side fragments that fall on the wrong AP side.

    Endpoint classification uses image axis 1: voxels anterior to the chiasm
    center are optic nerve, and voxels posterior to it are optic tract. When a
    thin nerve is locally disconnected, nearest-centerline propagation can pull
    that fragment onto the tract label. This pass corrects isolated fragments
    by comparing their axis-1 centroid to the chiasm centroid.
    """
    out = labels.copy()
    repaired: dict[str, int] = {}
    chiasm_pts = np.argwhere(out == 3)
    if chiasm_pts.size == 0:
        return out, repaired
    chiasm_y = float(np.mean(chiasm_pts[:, 1]))

    for nerve_label, tract_label in ((1, 4), (2, 5)):
        for source_label, target_label, direction in (
            (tract_label, nerve_label, "anterior"),
            (nerve_label, tract_label, "posterior"),
        ):
            component_img, n_components = ndi.label(out == source_label)
            changed = 0
            for component in range(1, n_components + 1):
                pts = np.argwhere(component_img == component)
                size = int(pts.shape[0])
                if size < min_component_voxels:
                    continue
                centroid_y = float(np.mean(pts[:, 1]))
                if direction == "anterior":
                    should_reassign = centroid_y > chiasm_y + margin_voxels
                else:
                    should_reassign = centroid_y < chiasm_y - margin_voxels
                if should_reassign:
                    out[component_img == component] = target_label
                    changed += size
            if changed:
                repaired[f"{source_label}_to_{target_label}_voxels"] = int(changed)
    return out, repaired


def fill_unassigned_foreground(labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    out = labels.copy()
    missing = mask & (out == 0)
    missing_count = int(np.count_nonzero(missing))
    if missing_count == 0:
        return out, 0
    if not np.any(out > 0):
        return out, missing_count

    _, nearest = ndi.distance_transform_edt(out == 0, return_indices=True)
    nearest_labels = out[nearest[0], nearest[1], nearest[2]]
    out[missing] = nearest_labels[missing]
    return out, missing_count


def split_mask(
    mask: np.ndarray,
    spacing: np.ndarray,
    min_component_voxels: int,
    chiasm_path_mm: float,
    compact_chiasm_core: bool,
    chiasm_core_radii_mm: np.ndarray,
    separate_pairs: bool,
    separator_iterations: int,
):
    clean = clean_mask(mask, min_component_voxels)
    if not np.any(clean):
        raise ValueError("Mask has no foreground after cleanup.")

    largest_scaffold, n_components, component_sizes = largest_component(clean)
    scaffold = clean
    skel = skeletonize_3d(scaffold) > 0
    degrees_img = skeleton_degrees(skel)
    coords = np.argwhere(skel)
    if coords.shape[0] < 5:
        raise ValueError("Skeleton is too small to split.")

    degrees = degrees_img[tuple(coords.T)]
    endpoint_indices = np.flatnonzero(degrees == 1)
    initial_source = infer_chiasm_source(coords, degrees)
    graph = build_position_graph(coords, spacing)
    dist, prev = dijkstra(graph, initial_source)
    endpoints = select_endpoints(coords, endpoint_indices, coords[initial_source], dist)
    if len(endpoints) < 4:
        raise ValueError(f"Could not identify four endpoint quadrants; found {sorted(endpoints)}")

    source = refine_chiasm_source_from_endpoints(coords, graph, endpoints)
    dist, prev = dijkstra(graph, source)
    endpoints = select_endpoints(coords, endpoint_indices, coords[source], dist)
    if len(endpoints) < 4:
        raise ValueError(f"Could not identify four endpoint quadrants after source refinement; found {sorted(endpoints)}")

    centerline_labels, path_lengths = make_centerline_labels(
        coords, source, prev, endpoints, spacing, chiasm_path_mm
    )
    out, centerline_img = assign_volume_from_centerline(clean, coords, centerline_labels)
    if compact_chiasm_core:
        out = apply_compact_chiasm_core(
            out,
            centerline_img,
            clean,
            coords[source],
            spacing,
            chiasm_core_radii_mm,
        )
    separator_converted_voxels = 0
    if separate_pairs:
        out, separator_converted_voxels = separate_nerve_tract_pairs(out, separator_iterations)
    out, repaired_pair_components = repair_missing_pair_components(
        out,
        min_component_voxels=min_component_voxels,
    )
    out, repaired_nerve_tract_orientation = repair_nerve_tract_orientation(out)
    out, filled_unassigned_voxels = fill_unassigned_foreground(out, clean)
    metadata = {
        "foreground_components": int(n_components),
        "foreground_component_sizes": component_sizes,
        "centerline_scaffold_voxels": int(np.count_nonzero(largest_scaffold)),
        "centerline_all_component_voxels": int(np.count_nonzero(scaffold)),
        "skeleton_voxels": int(coords.shape[0]),
        "endpoint_voxels": int(endpoint_indices.size),
        "initial_chiasm_source_voxel": "x".join(str(int(v)) for v in coords[initial_source]),
        "chiasm_source_voxel": "x".join(str(int(v)) for v in coords[source]),
        "selected_endpoints": {
            key: "x".join(str(int(v)) for v in coords[index]) for key, index in endpoints.items()
        },
        "compact_chiasm_core": bool(compact_chiasm_core),
        "chiasm_core_radii_mm": "x".join(f"{float(v):.3f}" for v in chiasm_core_radii_mm),
        "separate_nerve_tract_pairs": bool(separate_pairs),
        "separator_iterations": int(separator_iterations),
        "separator_converted_voxels": int(separator_converted_voxels),
        "repaired_pair_components": repaired_pair_components,
        "repaired_nerve_tract_orientation": repaired_nerve_tract_orientation,
        "filled_unassigned_voxels": int(filled_unassigned_voxels),
        **path_lengths,
    }
    return out, centerline_img, clean, metadata
