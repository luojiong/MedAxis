"""
Texture radiomics — GLCM(24), GLRLM(16), GLSZM(16), GLDM(14), NGTDM(5).

3D vectorized implementations following the standard (pyradiomics) formulas.
Input intensities are quantized to ``N`` levels (default 32).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# 13 3D directions (pyradiomics convention, without sign symmetry).
_DIRECTIONS_13 = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
]


def quantize(arr: np.ndarray, levels: int = 32) -> np.ndarray:
    """Quantize intensities to [0, levels-1] using the full range."""
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros(arr.shape, dtype=np.int16)
    q = np.floor((arr - lo) / (hi - lo) * levels)
    return np.clip(q, 0, levels - 1).astype(np.int16)


def _feature_map(name: str, value: float, family: str) -> dict:
    return {f"{family}_{name}": float(value)}


# ---------------------------------------------------------------------------
# GLCM — gray level co-occurrence matrix
# ---------------------------------------------------------------------------

def glcm_features(arr: np.ndarray, levels: int = 32, distances: tuple = (1,)) -> dict:
    """24 features: 6 properties x 4 direction groups (mean over directions)."""
    q = quantize(arr, levels)
    out: dict = {}
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm")
    # Per-direction group: axes (3 groups) + diagonals (combined) = 4 groups.
    groups = [
        [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(1, 1, 0), (1, -1, 0)],
        [(1, 0, 1), (1, 0, -1)],
        [(0, 1, 1), (0, 1, -1)],
        [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)],
    ]
    for gi, group in enumerate(groups):
        # Accumulate co-occurrence over the group's directions.
        p = np.zeros((levels, levels), dtype=np.float64)
        for dx, dy, dz in group:
            shifted = np.zeros_like(q)
            shifted = np.roll(q, shift=dx, axis=2)
            shifted = np.roll(shifted, shift=dy, axis=1)
            shifted = np.roll(shifted, shift=dz, axis=0)
            mask = np.ones(q.shape, dtype=bool)
            if dx > 0:
                mask[..., dx:] = False
            elif dx < 0:
                mask[..., :dx] = False
            if dy > 0:
                mask[:, dy:, :] = False
            elif dy < 0:
                mask[:, :dy, :] = False
            if dz > 0:
                mask[dz:, :, :] = False
            elif dz < 0:
                mask[:dz, :, :] = False
            pairs = np.stack([q[mask], shifted[mask]], axis=1)
            if len(pairs):
                np.add.at(p, (pairs[:, 0], pairs[:, 1]), 1.0)
        total = p.sum()
        if total <= 0:
            continue
        p /= total

        i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        diff = np.abs(i_idx - j_idx)
        contrast = float((p * diff**2).sum())
        dissimilarity = float((p * diff).sum())
        homogeneity = float((p / (1.0 + diff)).sum())
        asm = float((p**2).sum())
        energy = float(np.sqrt(asm))
        # Correlation (fall back to 0 when a row/col is constant).
        mi = (p * i_idx).sum()
        mj = (p * j_idx).sum()
        si = np.sqrt(max((p * (i_idx - mi) ** 2).sum(), 1e-12))
        sj = np.sqrt(max((p * (j_idx - mj) ** 2).sum(), 1e-12))
        correlation = float(((p * (i_idx - mi) * (j_idx - mj)).sum()) / (si * sj)) if si * sj > 1e-12 else 0.0

        for prop, value in zip(props, [contrast, dissimilarity, homogeneity, energy, correlation, asm]):
            out[f"glcm_{prop}_{gi}"] = value
    return out


# ---------------------------------------------------------------------------
# GLRLM — gray level run length matrix
# ---------------------------------------------------------------------------

def _run_length_matrix(q: np.ndarray, levels: int) -> tuple[np.ndarray, int]:
    """Build GLRLM over all 13 directions; returns (matrix, max_run)."""
    max_run = 0
    matrices = []
    for dx, dy, dz in _DIRECTIONS_13:
        # Walk along the direction, tracking runs of equal values.
        runs = []
        # Iterate over lines parallel to the direction.
        shape = q.shape
        # Number of lines = product of the other two dimensions.
        nk, nj, ni = shape
        for line_start in _line_starts(shape, (dx, dy, dz)):
            values = _sample_line(q, line_start, (dx, dy, dz))
            # Runs of identical values.
            if len(values) == 0:
                continue
            run_start = 0
            for pos in range(1, len(values) + 1):
                if pos == len(values) or values[pos] != values[run_start]:
                    runs.append((int(values[run_start]), pos - run_start))
                    run_start = pos
        if not runs:
            continue
        max_run = max(max_run, max(r for _, r in runs))
        m = np.zeros((levels, max_run + 1), dtype=np.float64)
        for value, length in runs:
            m[value, length] += 1.0
        matrices.append(m)
    if not matrices:
        return np.zeros((levels, 1)), 1
    out = np.zeros((levels, max_run + 1), dtype=np.float64)
    for m in matrices:
        if m.shape[1] < out.shape[1]:
            pad = np.zeros((levels, out.shape[1] - m.shape[1]))
            m = np.hstack([m, pad])
        out += m
    return out, max(1, max_run)


def _line_starts(shape, direction):
    """Starting indices of all lines parallel to `direction`."""
    nk, nj, ni = shape
    dx, dy, dz = direction
    starts = []
    # Choose the two axes not aligned with the direction as free coordinates.
    if dx != 0:
        for j in range(nj):
            for k in range(nk):
                starts.append((k, j, 0))
    elif dy != 0:
        for i in range(ni):
            for k in range(nk):
                starts.append((k, 0, i))
    else:
        for i in range(ni):
            for j in range(nj):
                starts.append((0, j, i))
    return starts


def _sample_line(q, start, direction):
    nk, nj, ni = q.shape
    dx, dy, dz = direction
    k, j, i = start
    values = []
    while 0 <= k < nk and 0 <= j < nj and 0 <= i < ni:
        values.append(q[k, j, i])
        k += dz
        j += dy
        i += dx
    return values


def glrlm_features(arr: np.ndarray, levels: int = 32) -> dict:
    q = quantize(arr, levels)
    m, max_run = _run_length_matrix(q, levels)
    nr = m.sum()
    if nr <= 0:
        return {}
    p = m / nr
    g_idx = np.arange(levels, dtype=np.float64)
    r_idx = np.arange(max_run + 1, dtype=np.float64)

    sre = float((p / (r_idx[None, :] ** 2 + 1e-12)).sum())
    lre = float((p * r_idx[None, :] ** 2).sum())
    gln = float((p.sum(axis=1) ** 2).sum())
    rln = float((p.sum(axis=0) ** 2).sum())
    rp = float(nr / max(q.size, 1))
    lg = float((p * g_idx[:, None]).sum())
    hg = float((p * g_idx[:, None] ** 2).sum())
    srlgle = float((p / (g_idx[:, None] ** 2 * r_idx[None, :] ** 2 + 1e-12)).sum())
    srhgle = float((p * g_idx[:, None] ** 2 / (r_idx[None, :] ** 2 + 1e-12)).sum())
    lrlgle = float((p * r_idx[None, :] ** 2 / (g_idx[:, None] ** 2 + 1e-12)).sum())
    lrhgle = float((p * g_idx[:, None] ** 2 * r_idx[None, :] ** 2).sum())
    glv = float((((p.sum(axis=1) - gln / levels) ** 2).sum() / max(levels - 1, 1)) if levels > 1 else 0.0)
    rlv = float((((p.sum(axis=0) - rln / (max_run + 1)) ** 2).sum() / max(max_run, 1)) if max_run > 0 else 0.0)
    rp_entropy = float(-(p * np.log(p + 1e-12)).sum())
    rp_energy = float((p**2).sum())
    names = ["SRE", "LRE", "GLN", "RLN", "RP", "LGRE", "HGRE",
             "SRLGLE", "SRHGLE", "LRLGLE", "LRHGLE", "GLV", "RLV", "RPE", "RPA"]
    values = [sre, lre, gln, rln, rp, lg, hg, srlgle, srhgle, lrlgle, lrhgle, glv, rlv, rp_entropy, rp_energy]
    return {f"glrlm_{n.lower()}": v for n, v in zip(names, values)}


# ---------------------------------------------------------------------------
# GLSZM — gray level size zone matrix
# ---------------------------------------------------------------------------

def glszm_features(arr: np.ndarray, levels: int = 32) -> dict:
    q = quantize(arr, levels)
    # Per intensity level, label connected components and count zone sizes.
    zones = []  # (level, size)
    for level in range(levels):
        mask = q == level
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=np.arange(1, n + 1))
        for s in sizes:
            zones.append((level, int(s)))
    if not zones:
        return {}
    max_zone = max(s for _, s in zones)
    m = np.zeros((levels, max_zone + 1), dtype=np.float64)
    for level, size in zones:
        m[level, size] += 1.0
    nz = m.sum()
    if nz <= 0:
        return {}
    p = m / nz
    g_idx = np.arange(levels, dtype=np.float64)
    z_idx = np.arange(max_zone + 1, dtype=np.float64)

    sae = float((p / (z_idx[None, :] ** 2 + 1e-12)).sum())
    lae = float((p * z_idx[None, :] ** 2).sum())
    gln = float((p.sum(axis=1) ** 2).sum())
    zn = float((p.sum(axis=0) ** 2).sum())
    zp = float(nz / max(q.size, 1))
    lgze = float((p * g_idx[:, None]).sum())
    hgze = float((p * g_idx[:, None] ** 2).sum())
    salgle = float((p / (g_idx[:, None] ** 2 * z_idx[None, :] ** 2 + 1e-12)).sum())
    sahgle = float((p * g_idx[:, None] ** 2 / (z_idx[None, :] ** 2 + 1e-12)).sum())
    lalgle = float((p * z_idx[None, :] ** 2 / (g_idx[:, None] ** 2 + 1e-12)).sum())
    lahgle = float((p * g_idx[:, None] ** 2 * z_idx[None, :] ** 2).sum())
    glv = float((((p.sum(axis=1) - gln / levels) ** 2).sum() / max(levels - 1, 1)) if levels > 1 else 0.0)
    zv = float((((p.sum(axis=0) - zn / (max_zone + 1)) ** 2).sum() / max(max_zone, 1)) if max_zone > 0 else 0.0)
    ze = float(-(p * np.log(p + 1e-12)).sum())
    za = float((p**2).sum())
    names = ["SAE", "LAE", "GLN", "ZN", "ZP", "LGZE", "HGZE",
             "SALGLE", "SAHGLE", "LALGLE", "LAHGLE", "GLV", "ZV", "ZE", "ZA"]
    values = [sae, lae, gln, zn, zp, lgze, hgze, salgle, sahgle, lalgle, lahgle, glv, zv, ze, za]
    return {f"glszm_{n.lower()}": v for n, v in zip(names, values)}


# ---------------------------------------------------------------------------
# GLDM — gray level dependence matrix
# ---------------------------------------------------------------------------

def gldm_features(arr: np.ndarray, levels: int = 32, delta: int = 0) -> dict:
    q = quantize(arr, levels)
    nk, nj, ni = q.shape
    # Dependence: count neighbors within Chebyshev radius 1 whose |diff| <= delta.
    padded = np.pad(q, 1, mode="edge").astype(np.float64)
    center = padded[1:-1, 1:-1, 1:-1]
    deps = np.zeros(q.shape, dtype=np.int16)
    for dk in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dk == dj == di == 0:
                    continue
                neighbor = padded[1 + dk:1 + dk + nk, 1 + dj:1 + dj + nj, 1 + di:1 + di + ni]
                deps += (np.abs(center - neighbor) <= delta).astype(np.int16)

    m = np.zeros((levels, 27), dtype=np.float64)  # dependence up to 26
    for level in range(levels):
        mask = q == level
        if mask.any():
            np.add.at(m[level], deps[mask], 1.0)
    nd = m.sum()
    if nd <= 0:
        return {}
    p = m / nd
    g_idx = np.arange(levels, dtype=np.float64)
    d_idx = np.arange(27, dtype=np.float64)

    sde = float((p / (d_idx[None, :] ** 2 + 1e-12)).sum())
    lde = float((p * d_idx[None, :] ** 2).sum())
    gln = float((p.sum(axis=1) ** 2).sum())
    dn = float((p.sum(axis=0) ** 2).sum())
    dp = float(nd / max(q.size, 1))
    lg = float((p * g_idx[:, None]).sum())
    hg = float((p * g_idx[:, None] ** 2).sum())
    sdlgle = float((p / (g_idx[:, None] ** 2 * d_idx[None, :] ** 2 + 1e-12)).sum())
    sdhgle = float((p * g_idx[:, None] ** 2 / (d_idx[None, :] ** 2 + 1e-12)).sum())
    ldlgle = float((p * d_idx[None, :] ** 2 / (g_idx[:, None] ** 2 + 1e-12)).sum())
    ldhgle = float((p * g_idx[:, None] ** 2 * d_idx[None, :] ** 2).sum())
    glv = float((((p.sum(axis=1) - gln / levels) ** 2).sum() / max(levels - 1, 1)) if levels > 1 else 0.0)
    dv = float((((p.sum(axis=0) - dn / 27) ** 2).sum() / 26.0) if 26 > 0 else 0.0)
    de = float(-(p * np.log(p + 1e-12)).sum())
    names = ["SDE", "LDE", "GLN", "DN", "DP", "LGRE", "HGRE",
             "SDLGE", "SDHGE", "LDLGE", "LDHGE", "GLV", "DV", "DE"]
    values = [sde, lde, gln, dn, dp, lg, hg, sdlgle, sdhgle, ldlgle, ldhgle, glv, dv, de]
    return {f"gldm_{n.lower()}": v for n, v in zip(names, values)}


# ---------------------------------------------------------------------------
# NGTDM — neighbourhood gray tone difference matrix
# ---------------------------------------------------------------------------

def ngtdm_features(arr: np.ndarray, levels: int = 32) -> dict:
    q = quantize(arr, levels).astype(np.float64)
    nk, nj, ni = q.shape
    padded = np.pad(q, 1, mode="edge")
    center = padded[1:-1, 1:-1, 1:-1]

    # Mean of the 26-neighborhood.
    s = np.zeros(q.shape, dtype=np.float64)
    count = np.zeros(q.shape, dtype=np.float64)
    for dk in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dk == dj == di == 0:
                    continue
                s += padded[1 + dk:1 + dk + nk, 1 + dj:1 + dj + nj, 1 + di:1 + di + ni]
                count += 1.0
    neighbor_mean = s / count

    # p_i: probability of gray level i over the volume.
    levels_arr = np.arange(levels, dtype=np.float64)
    hist = np.bincount(q.ravel().astype(np.int64), minlength=levels).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return {}
    p_i = hist / total

    # s_i: sum of |i - mean_neighbor| over voxels with gray level i.
    diff = np.abs(center - neighbor_mean)
    s_i = np.zeros(levels, dtype=np.float64)
    n_i = np.zeros(levels, dtype=np.float64)
    for level in range(levels):
        mask = q.astype(np.int64) == level
        n_i[level] = mask.sum()
        if n_i[level] > 0:
            s_i[level] = diff[mask].sum()

    nonzero = p_i > 0
    n_minus = max(total - s_i[nonzero].sum(), 1e-12)

    coarseness = float(1.0 / (s_i[nonzero].sum() + 1e-12))
    contrast = float(
        (p_i[nonzero, None] * p_i[None, nonzero] * np.abs(levels_arr[nonzero, None] - levels_arr[None, nonzero])).sum()
        / n_minus * (1.0 / max(len(levels_arr[nonzero]), 1)))
    busyness = float(
        (p_i[nonzero] * s_i[nonzero]).sum()
        / (p_i[None, nonzero] * p_i[nonzero, None] * np.abs(levels_arr[None, nonzero] - levels_arr[nonzero, None])).sum())
    complexity = float(
        (np.abs(levels_arr[None, nonzero] - levels_arr[nonzero, None])
         * (p_i[None, nonzero] * s_i[nonzero, None] + p_i[nonzero, None] * s_i[None, nonzero])
         / (p_i[nonzero, None] + p_i[None, nonzero] + 1e-12)).sum())
    strength = float(
        (p_i[nonzero] * s_i[nonzero]).sum()
        / (np.abs(levels_arr[None, nonzero] - levels_arr[nonzero, None]).sum() + 1e-12))
    names = ["coarseness", "contrast", "busyness", "complexity", "strength"]
    values = [coarseness, contrast, busyness, complexity, strength]
    return {f"ngtdm_{n}": v for n, v in zip(names, values)}


def extract_texture_features(arr: np.ndarray, families=("glcm", "glrlm", "glszm", "gldm", "ngtdm"),
                             levels: int = 32) -> dict:
    """Run the requested texture feature families on a 3D volume."""
    out: dict = {}
    if "glcm" in families:
        out.update(glcm_features(arr, levels))
    if "glrlm" in families:
        out.update(glrlm_features(arr, levels))
    if "glszm" in families:
        out.update(glszm_features(arr, levels))
    if "gldm" in families:
        out.update(gldm_features(arr, levels))
    if "ngtdm" in families:
        out.update(ngtdm_features(arr, levels))
    return out
