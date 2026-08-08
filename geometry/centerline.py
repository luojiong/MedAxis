"""
Vascular centerline extraction — 3D skeletonization + graph shortest path.

Pipeline (blueprint Phase 6):
  1. 3D binary thinning (Lee's algorithm via scikit-image) of a vessel label
  2. skeleton voxel graph (26-connectivity), endpoints detected
  3. Dijkstra shortest path between the two requested endpoints (or the
     farthest pair) → centerline polyline in world coordinates
  4. B-spline fit + Frenet frames via the native medaxis_occ module
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def skeletonize_3d(label: np.ndarray) -> np.ndarray:
    """Lee's 3D thinning of a binary label volume."""
    from skimage.morphology import skeletonize

    binary = np.asarray(label) > 0
    return skeletonize(binary).astype(np.uint8)


def _neighbors_26():
    nbs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nbs.append((dx, dy, dz))
    return nbs


def skeleton_graph(skeleton: np.ndarray):
    """Build an adjacency list of skeleton voxels (26-connectivity)."""
    nbs = _neighbors_26()
    indices = np.argwhere(skeleton > 0)
    index_of = {tuple(v): i for i, v in enumerate(indices)}
    adjacency = [[] for _ in indices]
    for i, (k, j, i_) in enumerate(indices):
        for dx, dy, dz in nbs:
            nb = (k + dz, j + dy, i_ + dx)
            if nb in index_of:
                adjacency[i].append(index_of[nb])
    return indices, adjacency


def find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Voxels with exactly one skeleton neighbor (curve ends)."""
    nbs = _neighbors_26()
    skeleton_bool = skeleton > 0
    endpoints = []
    for (k, j, i) in np.argwhere(skeleton_bool):
        count = 0
        for dx, dy, dz in nbs:
            k2, j2, i2 = k + dz, j + dy, i + dx
            if 0 <= k2 < skeleton.shape[0] and 0 <= j2 < skeleton.shape[1] \
                    and 0 <= i2 < skeleton.shape[2] and skeleton_bool[k2, j2, i2]:
                count += 1
        if count <= 1:
            endpoints.append((k, j, i))
    return np.array(endpoints) if endpoints else np.zeros((0, 3), dtype=int)


def _dijkstra(adjacency, start: int, end: int) -> Optional[list[int]]:
    import heapq

    dist = {start: 0}
    prev = {}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == end:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v in adjacency[u]:
            nd = d + 1
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if end not in dist:
        return None
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def extract_centerline(
    label: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    endpoints: Optional[list] = None,
) -> dict:
    """Extract the vessel centerline of a binary label.

    Returns:
        dict with ``points`` (Nx3 world coordinates), ``skeleton`` (voxel
        indices), ``endpoints`` and ``length_mm``.
    """
    skeleton = skeletonize_3d(label)
    indices, adjacency = skeleton_graph(skeleton)
    if len(indices) == 0:
        return {"points": np.zeros((0, 3)), "skeleton": skeleton,
                "endpoints": np.zeros((0, 3)), "length_mm": 0.0}

    ends = find_endpoints(skeleton)
    if endpoints is not None and len(endpoints) == 2:
        start = tuple(int(v) for v in endpoints[0])
        stop = tuple(int(v) for v in endpoints[1])
        index_of = {tuple(v): i for i, v in enumerate(indices)}
        if start in index_of and stop in index_of:
            path = _dijkstra(adjacency, index_of[start], index_of[stop])
        else:
            path = None
    else:
        # Farthest endpoint pair by Euclidean distance.
        path = None
        best = -1.0
        best_pair = None
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                d = float(np.linalg.norm(ends[a] - ends[b]))
                if d > best:
                    best = d
                    best_pair = (a, b)
        if best_pair is not None:
            index_of = {tuple(v): i for i, v in enumerate(indices)}
            pa = _dijkstra(adjacency, index_of[tuple(ends[best_pair[0]])],
                           index_of[tuple(ends[best_pair[1]])])
            if pa is not None:
                path = pa

    if path is None:
        # Degenerate: use the whole skeleton ordering.
        path = list(range(len(indices)))

    voxel_points = indices[path]
    world = np.asarray(voxel_points, dtype=float) * np.asarray(spacing) + np.asarray(origin)
    seg_lengths = np.linalg.norm(np.diff(world, axis=0), axis=1)
    return {
        "points": world,
        "skeleton": skeleton,
        "endpoints": ends,
        "length_mm": float(seg_lengths.sum()),
    }


def centerline_to_bspline(points: np.ndarray, num_samples: int = 200) -> dict:
    """Fit the centerline with a B-spline and sample Frenet frames (native)."""
    from core.native_extensions import get_native_module

    occ = get_native_module("medaxis_occ")
    if occ is None or len(points) < 4:
        return {"points": points, "frames": None}
    control = [list(map(float, p)) for p in points]
    samples = occ.sample_curve_equal_arc_length(control, num_samples)
    parameters = [i / (num_samples - 1) for i in range(num_samples)]
    frames = occ.frenet_frames(control, parameters)
    return {
        "points": np.asarray(samples, dtype=float),
        "frames": frames,
        "length_mm": float(occ.curve_arc_length(control)),
    }
