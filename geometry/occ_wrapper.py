"""
Geometry layer — wrappers for OpenCASCADE and CGAL operations.
"""
import numpy as np

from core.native_extensions import get_native_module


class BSplineCurve:
    """Wrapper around OpenCASCADE Geom_BSplineCurve for CPR path definition."""

    def __init__(self, control_points: np.ndarray):
        """
        Args:
            control_points: Nx3 array of 3D control points
        """
        self.control_points = control_points

    @property
    def degree(self) -> int:
        return max(1, len(self.control_points) - 1)

    def sample_equal_arc_length(self, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample curve at equal arc-length intervals.
        Returns (points, tangents, normals) — each Nx3.
        Falls back to chord-length sampling if OCC unavailable.
        """
        pts = self.control_points
        if len(pts) < 2:
            return pts, np.zeros_like(pts), np.zeros_like(pts)

        native = get_native_module("medaxis_occ")
        if native is not None:
            try:
                control_points = [tuple(map(float, point)) for point in np.asarray(pts, dtype=float)]
                sampled = np.asarray(
                    native.sample_curve_equal_arc_length(control_points, int(num_samples)),
                    dtype=float,
                )
                parameters = np.linspace(0.0, 1.0, len(sampled))
                frames = native.frenet_frames(control_points, parameters.tolist())
                tangents = np.asarray([frame.tangent for frame in frames], dtype=float)
                normals = np.asarray([frame.normal for frame in frames], dtype=float)
                return sampled, tangents, normals
            except (RuntimeError, ValueError, TypeError):
                # Keep geometry usable when a native build has incompatible input.
                pass

        # Chord-length parameterization
        chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        chord_lengths = np.insert(np.cumsum(chords), 0, 0)
        total = chord_lengths[-1]
        if total < 1e-10:
            return np.tile(pts[0], (num_samples, 1)), np.zeros((num_samples, 3)), np.zeros((num_samples, 3))

        t_samples = np.linspace(0, total, num_samples)
        sampled_points = np.zeros((num_samples, 3))
        tangents = np.zeros((num_samples, 3))
        normals = np.zeros((num_samples, 3))

        for i, t in enumerate(t_samples):
            idx = np.searchsorted(chord_lengths, t, side='right') - 1
            idx = max(0, min(idx, len(pts) - 2))
            local_t = (t - chord_lengths[idx]) / max(1e-10, chord_lengths[idx + 1] - chord_lengths[idx])
            local_t = max(0.0, min(1.0, local_t))
            sampled_points[i] = pts[idx] * (1 - local_t) + pts[idx + 1] * local_t
            tangents[i] = pts[idx + 1] - pts[idx]
            tangents[i] /= max(1e-10, np.linalg.norm(tangents[i]))

        # Compute normals (Frenet-Serret approximation)
        for i in range(num_samples):
            t_vec = tangents[i]
            if i == 0:
                ref = np.array([0, 0, 1]) if abs(t_vec[2]) < 0.9 else np.array([1, 0, 0])
            else:
                ref = normals[i - 1]
            n = np.cross(t_vec, ref)
            if np.linalg.norm(n) < 1e-10:
                n = np.cross(t_vec, np.array([0, 1, 0]))
            n /= max(1e-10, np.linalg.norm(n))
            normals[i] = n

        return sampled_points, tangents, normals

    def compute_frenet_frame(self, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Frenet-Serret frame (tangent, normal, binormal) along the curve."""
        pts, tangents, normals = self.sample_equal_arc_length(num_samples)
        binormals = np.cross(tangents, normals)
        for i in range(len(binormals)):
            b = binormals[i]
            if np.linalg.norm(b) > 1e-10:
                binormals[i] /= np.linalg.norm(b)
        return tangents, normals, binormals
