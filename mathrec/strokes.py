"""
Stroke preprocessing and feature extraction.

A *stroke* is one pen-down..pen-up trace: a list of points. We accept the
exact shape the frontend sends - a list of {"x":.., "y":.., "t":..} dicts -
as well as plain (x, y) tuples, and convert everything to numpy arrays.

The key idea (see package docstring): a symbol is described by the geometry
and direction of its trajectory, not by a resized bitmap. The feature
vector below deliberately keeps aspect ratio, curvature and stroke
direction, because those are exactly what separate '(' ')' '1' '/' '-'.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Conversion helpers
# ----------------------------------------------------------------------------
def stroke_to_array(stroke) -> np.ndarray:
    """Convert one stroke (list of {x,y[,t]} or (x,y)) to an Nx2 float array."""
    pts = []
    for p in stroke:
        if isinstance(p, dict):
            pts.append((float(p.get("x", 0.0)), float(p.get("y", 0.0))))
        else:
            pts.append((float(p[0]), float(p[1])))
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def stroke_times(stroke) -> np.ndarray:
    """Return the timestamp array for a stroke, or an empty array."""
    ts = []
    for p in stroke:
        if isinstance(p, dict) and "t" in p:
            ts.append(float(p["t"]))
    return np.asarray(ts, dtype=float) if len(ts) == len(stroke) else np.zeros((0,))


def clean_strokes(strokes) -> List[np.ndarray]:
    """Drop empty/degenerate strokes; return list of Nx2 arrays (>=1 point)."""
    out = []
    for s in strokes or []:
        if not isinstance(s, (list, tuple)):
            continue
        arr = stroke_to_array(s)
        if arr.shape[0] >= 1:
            out.append(arr)
    return out


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def bbox(xy: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) for an Nx2 array."""
    if xy.shape[0] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1 = xy[:, 0].min(), xy[:, 1].min()
    x2, y2 = xy[:, 0].max(), xy[:, 1].max()
    return (float(x1), float(y1), float(x2), float(y2))


def group_bbox(group: Sequence[np.ndarray]) -> Tuple[float, float, float, float]:
    """Bounding box over a list of strokes."""
    pts = np.vstack([s for s in group if s.shape[0] > 0]) if group else np.zeros((0, 2))
    return bbox(pts)


def arc_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    d = np.diff(xy, axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def resample(xy: np.ndarray, n: int = 24) -> np.ndarray:
    """Resample a polyline to n points equally spaced by arc length.

    This removes the effect of drawing speed / sampling rate, so the same
    shape drawn fast or slow yields the same feature vector.
    """
    if xy.shape[0] == 0:
        return np.zeros((n, 2), dtype=float)
    if xy.shape[0] == 1:
        return np.repeat(xy, n, axis=0)

    d = np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total <= 1e-9:
        return np.repeat(xy[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=float)
    out[:, 0] = np.interp(targets, cum, xy[:, 0])
    out[:, 1] = np.interp(targets, cum, xy[:, 1])
    return out


def normalize_unit(xy: np.ndarray) -> np.ndarray:
    """Center on centroid and scale by the larger dimension (aspect preserved)."""
    if xy.shape[0] == 0:
        return xy
    c = xy.mean(axis=0)
    centered = xy - c
    span = max(np.ptp(centered[:, 0]), np.ptp(centered[:, 1]), 1e-6)
    return centered / span


# ----------------------------------------------------------------------------
# Shape descriptors
# ----------------------------------------------------------------------------
def direction_histogram(xy: np.ndarray, bins: int = 8) -> np.ndarray:
    """Histogram of movement directions along the trajectory (rotation cue)."""
    hist = np.zeros(bins, dtype=float)
    if xy.shape[0] < 2:
        return hist
    d = np.diff(xy, axis=0)
    seg_len = np.sqrt((d ** 2).sum(axis=1))
    angles = np.arctan2(d[:, 1], d[:, 0])  # -pi..pi
    for ang, w in zip(angles, seg_len):
        idx = int(((ang + math.pi) / (2 * math.pi)) * bins) % bins
        hist[idx] += w
    s = hist.sum()
    return hist / s if s > 0 else hist


def total_turning(xy: np.ndarray) -> float:
    """Sum of absolute turning angles (curviness). Lines ~0, loops ~2*pi."""
    if xy.shape[0] < 3:
        return 0.0
    d = np.diff(xy, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    da = np.diff(ang)
    da = (da + math.pi) % (2 * math.pi) - math.pi  # wrap to -pi..pi
    return float(np.abs(da).sum())


def signed_curvature(xy: np.ndarray) -> float:
    """Net signed turning. Positive vs negative tells '(' from ')'."""
    if xy.shape[0] < 3:
        return 0.0
    d = np.diff(xy, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    da = np.diff(ang)
    da = (da + math.pi) % (2 * math.pi) - math.pi
    return float(da.sum())


# ----------------------------------------------------------------------------
# Per-symbol feature vector
# ----------------------------------------------------------------------------
N_TRAJ = 20          # resampled points per group used for the trajectory code
FEATURE_DIM = N_TRAJ * 2 + 8 + 5   # traj(40) + dir-hist(8) + scalars(5) = 53


def group_to_points(group: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate a group's strokes into one Nx2 trace, in a canonical order.

    Strokes are ordered by their top-left corner (x then y). Using one fixed
    order in BOTH training and inference is essential: segmentation reorders
    strokes spatially, so if the feature builder used drawing order instead,
    multi-stroke symbols (e.g. '+') would produce a different trajectory at
    inference than during training and be misclassified.
    """
    parts = [s for s in group if s.shape[0] > 0]
    if not parts:
        return np.zeros((0, 2), dtype=float)
    parts = sorted(parts, key=lambda s: (float(s[:, 0].min()), float(s[:, 1].min())))
    return np.vstack(parts)


def feature_vector(group: Sequence[np.ndarray]) -> np.ndarray:
    """Build the fixed-length feature vector used by the classifier.

    Layout:
        [0          : 2*N_TRAJ)  normalized resampled trajectory (x,y interleaved)
        [2*N_TRAJ   : +8)        direction histogram
        last 5 scalars:          aspect ratio code, stroke count code,
                                  total turning, signed curvature, fill ratio
    """
    pts = group_to_points(group)
    if pts.shape[0] == 0:
        return np.zeros(FEATURE_DIM, dtype=float)

    x1, y1, x2, y2 = bbox(pts)
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)

    res = resample(pts, N_TRAJ)
    norm = normalize_unit(res)
    traj = norm.reshape(-1)  # 2*N_TRAJ

    dh = direction_histogram(res, bins=8)

    aspect = math.log((w / h)) if h > 0 else 0.0          # >0 wide, <0 tall
    nstroke = min(len(group), 4) / 4.0
    turning = total_turning(res) / (2 * math.pi)
    signed = signed_curvature(res) / (2 * math.pi)
    # fill ratio: arc length vs bbox diagonal (lines small, loops large)
    diag = math.hypot(w, h)
    fill = arc_length(pts) / diag if diag > 0 else 0.0

    scalars = np.array([aspect, nstroke, turning, signed, min(fill, 4.0)], dtype=float)
    return np.concatenate([traj, dh, scalars]).astype(float)


# ----------------------------------------------------------------------------
# Geometric summary used by segmentation + layout (cheap, no resampling)
# ----------------------------------------------------------------------------
class GroupGeom:
    """Lightweight geometry of a symbol group for segmentation / layout."""

    __slots__ = ("x1", "y1", "x2", "y2", "cx", "cy", "w", "h", "n_strokes")

    def __init__(self, group: Sequence[np.ndarray]):
        self.x1, self.y1, self.x2, self.y2 = group_bbox(group)
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0
        self.n_strokes = len(group)

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h > 1e-6 else 999.0
