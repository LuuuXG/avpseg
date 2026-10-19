"""Measure optic nerve centerlines, surfaces, and component gaps."""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, mesh_surface_area

try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    from skimage.morphology import skeletonize

    def skeletonize_3d(image):
        return skeletonize(image, method="lee")


def neighbor_offsets() -> list[tuple[int, int, int]]:
    offsets: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


OFFSETS = neighbor_offsets()


def build_graph(coords: np.ndarray, spacing: np.ndarray) -> list[list[tuple[int, float]]]:
    coord_to_index = {tuple(coord.tolist()): i for i, coord in enumerate(coords)}
    graph: list[list[tuple[int, float]]] = [[] for _ in range(len(coords))]
    for i, coord in enumerate(coords):
        for offset in OFFSETS:
            nb = (int(coord[0] + offset[0]), int(coord[1] + offset[1]), int(coord[2] + offset[2]))
            j = coord_to_index.get(nb)
            if j is None:
                continue
            graph[i].append((j, float(np.linalg.norm(np.asarray(offset, dtype=float) * spacing))))
    return graph


def graph_components(graph: list[list[tuple[int, float]]]) -> list[list[int]]:
    seen = np.zeros(len(graph), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(graph)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nb, _ in graph[node]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        components.append(component)
    return components


def bridge_graph_components(
    graph: list[list[tuple[int, float]]],
    coords: np.ndarray,
    spacing: np.ndarray,
) -> list[list[tuple[int, float]]]:
    """Connect disconnected skeleton components with virtual Euclidean edges."""
    components = graph_components(graph)
    if len(components) <= 1:
        return graph

    graph = [neighbors.copy() for neighbors in graph]
    remaining = [np.asarray(component, dtype=int) for component in components]
    connected = remaining.pop(0)
    while remaining:
        tree = cKDTree(coords[connected].astype(float) * spacing)
        best_idx = 0
        best_connected_node = 0
        best_other_node = 0
        best_distance = np.inf
        for idx, component in enumerate(remaining):
            distances, nearest = tree.query(coords[component].astype(float) * spacing, k=1)
            local = int(np.argmin(distances))
            distance = float(distances[local])
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
                best_connected_node = int(connected[int(nearest[local])])
                best_other_node = int(component[local])
        graph[best_connected_node].append((best_other_node, best_distance))
        graph[best_other_node].append((best_connected_node, best_distance))
        connected = np.concatenate([connected, remaining.pop(best_idx)])
    return graph


def build_position_graph(coords: np.ndarray, spacing: np.ndarray) -> list[list[tuple[int, float]]]:
    return bridge_graph_components(build_graph(coords, spacing), coords, spacing)


def dijkstra(graph: list[list[tuple[int, float]]], source: int) -> np.ndarray:
    dist = np.full(len(graph), np.inf, dtype=float)
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
                heapq.heappush(heap, (nd, nb))
    return dist


def skeleton_degrees(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def bridge_mask_components(
    mask: np.ndarray,
    spacing: np.ndarray,
    max_gap_mm: float = 6.0,
) -> tuple[np.ndarray, int]:
    """Connect short component gaps for metric extraction without changing output masks."""
    out = mask.astype(bool).copy()
    bridge_voxels = 0
    structure = np.ones((3, 3, 3), dtype=bool)
    component_img, n_components = ndi.label(out, structure=structure)
    while n_components > 1:
        previous_components = n_components
        previous_out = out.copy()
        components = [np.argwhere(component_img == idx) for idx in range(1, n_components + 1)]
        connected = components.pop(0)
        best_distance = np.inf
        best_pair: tuple[np.ndarray, np.ndarray] | None = None
        tree = cKDTree(connected.astype(float) * spacing)
        for component in components:
            distances, nearest = tree.query(component.astype(float) * spacing, k=1)
            local = int(np.argmin(distances))
            distance = float(distances[local])
            if distance < best_distance:
                best_distance = distance
                best_pair = (connected[int(nearest[local])], component[local])
        if best_pair is None or best_distance > max_gap_mm:
            break

        a, b = best_pair
        steps = int(np.max(np.abs(b - a))) + 1
        line = np.rint(np.linspace(a, b, steps)).astype(int)
        before = int(np.count_nonzero(out))
        out[tuple(line.T)] = True
        component_img, n_components = ndi.label(out, structure=structure)
        if n_components >= previous_components:
            out = previous_out
            break
        bridge_voxels += int(np.count_nonzero(out) - before)
    return out, bridge_voxels


def centerline_length_mm(mask: np.ndarray, spacing: np.ndarray) -> tuple[float, int, int]:
    skel = skeletonize_3d(mask) > 0
    coords = np.argwhere(skel)
    if coords.shape[0] < 2:
        return 0.0, int(coords.shape[0]), 0
    graph = build_position_graph(coords, spacing)
    degrees = skeleton_degrees(skel)[tuple(coords.T)]
    endpoints = np.flatnonzero(degrees == 1)
    candidates = endpoints if endpoints.size >= 2 else np.arange(coords.shape[0])
    best = 0.0
    for source in candidates:
        dist = dijkstra(graph, int(source))
        finite = dist[candidates][np.isfinite(dist[candidates])]
        if finite.size:
            best = max(best, float(finite.max()))
    return best, int(coords.shape[0]), int(endpoints.size)


def centerline_positions_mm(skel: np.ndarray, spacing: np.ndarray) -> dict[tuple[int, int, int], float]:
    """Return geodesic positions from the posterior/chiasm-side endpoint.

    Disconnected skeleton components are connected by virtual Euclidean edges so
    thin gaps still contribute to position ordering and centerline length.
    """
    coords = np.argwhere(skel)
    if coords.shape[0] == 0:
        return {}
    if coords.shape[0] == 1:
        return {tuple(coords[0].tolist()): 0.0}
    graph = build_position_graph(coords, spacing)
    degrees = skeleton_degrees(skel)[tuple(coords.T)]
    endpoints = np.flatnonzero(degrees == 1)
    if endpoints.size:
        source = int(endpoints[np.argmin(coords[endpoints, 1])])
    else:
        source = int(np.argmin(coords[:, 1]))
    dist = dijkstra(graph, source)
    finite = np.isfinite(dist)
    if not np.any(finite):
        dist[:] = 0.0
    return {tuple(coord.tolist()): float(pos) for coord, pos in zip(coords, dist)}


def surface_area_mm2(mask: np.ndarray, spacing: np.ndarray) -> float:
    if np.count_nonzero(mask) < 4:
        return 0.0
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(spacing))
    return float(mesh_surface_area(verts, faces))


def crop_to_mask(mask: np.ndarray, margin: int = 2) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return mask, np.zeros(3, dtype=int)
    start = np.maximum(pts.min(axis=0) - margin, 0)
    stop = np.minimum(pts.max(axis=0) + margin + 1, mask.shape)
    return mask[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]], start
