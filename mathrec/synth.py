"""
Synthetic stroke generator.

Produces handwriting-like stroke samples for each class so the classifier
can be trained with no downloaded dataset. Each class has a parametric
template (built from line / arc primitives in a unit box, y pointing down);
samples are produced by applying random affine warps, jitter and resampling.

This is the *baseline* data source. For production accuracy on real student
handwriting, train the same classifier on real strokes (collected from the
canvas) or on CROHME's online InkML traces - see train_strokes.py.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np

from .labels import CLASSES


# ----------------------------------------------------------------------------
# Primitives (unit box, y down)
# ----------------------------------------------------------------------------
def _line(p0, p1, n=12):
    t = np.linspace(0, 1, n)
    return np.stack([p0[0] + (p1[0] - p0[0]) * t,
                     p0[1] + (p1[1] - p0[1]) * t], axis=1)


def _arc(cx, cy, rx, ry, a0, a1, n=24):
    t = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)


# ----------------------------------------------------------------------------
# Templates: each returns a list of strokes (Nx2 arrays) in [0,1]x[0,1]
# ----------------------------------------------------------------------------
def t_0():
    return [_arc(0.5, 0.5, 0.32, 0.46, 0, 2 * math.pi, 36)]


def t_1():
    return [np.vstack([_line((0.32, 0.18), (0.5, 0.04), 5),
                       _line((0.5, 0.04), (0.5, 0.96), 16)])]


def t_2():
    top = _arc(0.5, 0.30, 0.28, 0.22, math.pi, 2 * math.pi + 0.4, 22)
    diag = _line((0.74, 0.42), (0.24, 0.92), 12)
    base = _line((0.24, 0.92), (0.78, 0.92), 10)
    return [np.vstack([top, diag, base])]


def t_3():
    a = _arc(0.46, 0.28, 0.26, 0.22, math.pi, 2.2 * math.pi, 18)
    b = _arc(0.46, 0.70, 0.28, 0.24, -0.9 * math.pi, 0.9 * math.pi, 20)
    return [np.vstack([a, b])]


def t_4():
    s1 = np.vstack([_line((0.62, 0.04), (0.20, 0.64), 12),
                    _line((0.20, 0.64), (0.84, 0.64), 12)])
    s2 = _line((0.62, 0.10), (0.62, 0.96), 14)
    return [s1, s2]


def t_5():
    top = _line((0.70, 0.06), (0.30, 0.06), 8)
    down = _line((0.30, 0.06), (0.30, 0.46), 8)
    bowl = _arc(0.46, 0.66, 0.30, 0.28, -0.6 * math.pi, 0.85 * math.pi, 20)
    return [np.vstack([top, down, bowl])]


def t_6():
    tail = _arc(0.55, 0.42, 0.30, 0.40, -0.45 * math.pi, math.pi, 20)
    loop = _arc(0.46, 0.70, 0.27, 0.24, 0, 2 * math.pi, 24)
    return [np.vstack([tail, loop])]


def t_7():
    return [np.vstack([_line((0.20, 0.08), (0.80, 0.08), 10),
                       _line((0.80, 0.08), (0.40, 0.96), 14)])]


def t_8():
    top = _arc(0.5, 0.30, 0.24, 0.22, 0, 2 * math.pi, 22)
    bot = _arc(0.5, 0.72, 0.28, 0.24, 0, 2 * math.pi, 24)
    return [np.vstack([top, bot])]


def t_9():
    loop = _arc(0.50, 0.32, 0.26, 0.24, 0, 2 * math.pi, 24)
    tail = _line((0.74, 0.32), (0.52, 0.96), 12)
    return [np.vstack([loop, tail])]


def t_plus():
    return [_line((0.5, 0.12), (0.5, 0.88), 12),
            _line((0.14, 0.5), (0.86, 0.5), 12)]


def t_minus():
    return [_line((0.10, 0.5), (0.90, 0.5), 14)]


def t_times():
    return [_line((0.18, 0.18), (0.82, 0.82), 12),
            _line((0.82, 0.18), (0.18, 0.82), 12)]


def t_divide():
    return [_line((0.12, 0.5), (0.88, 0.5), 12),
            _arc(0.5, 0.18, 0.05, 0.05, 0, 2 * math.pi, 8),
            _arc(0.5, 0.82, 0.05, 0.05, 0, 2 * math.pi, 8)]


def t_equals():
    return [_line((0.12, 0.38), (0.88, 0.38), 12),
            _line((0.12, 0.62), (0.88, 0.62), 12)]


def t_lparen():
    # opens to the right: a left-bending arc
    return [_arc(0.70, 0.5, 0.42, 0.48, 0.62 * math.pi, 1.38 * math.pi, 24)]


def t_rparen():
    # opens to the left
    return [_arc(0.30, 0.5, 0.42, 0.48, -0.38 * math.pi, 0.38 * math.pi, 24)]


def t_dot():
    return [_arc(0.5, 0.78, 0.04, 0.04, 0, 2 * math.pi, 8)]


def t_slash():
    return [_line((0.78, 0.08), (0.22, 0.92), 16)]


TEMPLATES: Dict[str, Callable[[], List[np.ndarray]]] = {
    "0": t_0, "1": t_1, "2": t_2, "3": t_3, "4": t_4,
    "5": t_5, "6": t_6, "7": t_7, "8": t_8, "9": t_9,
    "plus": t_plus, "minus": t_minus, "times": t_times,
    "divide": t_divide, "equals": t_equals,
    "lparen": t_lparen, "rparen": t_rparen, "dot": t_dot, "slash": t_slash,
}


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
def _augment(strokes: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
    # random rotation, shear, anisotropic scale, translation, jitter
    theta = rng.normal(0, 0.10)
    shear = rng.normal(0, 0.10)
    sx = rng.uniform(0.80, 1.20)
    sy = rng.uniform(0.80, 1.20)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Sh = np.array([[1.0, shear], [0.0, 1.0]])
    Sc = np.array([[sx, 0.0], [0.0, sy]])
    M = R @ Sh @ Sc

    tx, ty = rng.normal(0, 0.03, size=2)
    out = []
    for st in strokes:
        pts = (st - 0.5) @ M.T + 0.5
        pts = pts + np.array([tx, ty])
        jitter = rng.normal(0, 0.012, size=pts.shape)
        pts = pts + jitter
        # scale to a plausible pixel size and random position
        out.append(pts)
    # global scale to pixels
    scale = rng.uniform(40, 120)
    ox, oy = rng.uniform(0, 200), rng.uniform(0, 200)
    out = [p * scale + np.array([ox, oy]) for p in out]
    return out


def sample(label: str, rng: np.random.Generator) -> List[np.ndarray]:
    strokes = TEMPLATES[label]()
    return _augment(strokes, rng)


def make_dataset(per_class: int = 400, seed: int = 0):
    """Return (X features, y labels) for all classes."""
    from .strokes import feature_vector

    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, label in enumerate(CLASSES):
        for _ in range(per_class):
            strokes = sample(label, rng)
            X.append(feature_vector(strokes))
            y.append(ci)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)
