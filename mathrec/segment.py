"""
Stroke segmentation: group raw strokes into symbol candidates.

Segmentation is where the old pipeline's multi-symbol errors began: one
mis-grouped stroke corrupts everything downstream. We group conservatively
using bounding-box overlap, horizontal gap (relative to the median symbol
width) and stroke timing, while protecting two important cases:

  * multi-stroke symbols ('=', 'x', the division sign, '+') must be merged
    even though they are separate pen strokes;
  * a superscript/exponent must NOT be merged into its base.

The recognizer keeps these as *candidates*; the layout stage can still
re-interpret spatial relationships afterwards.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .strokes import GroupGeom, group_bbox


def _bbox_of(strokes: Sequence[np.ndarray]):
    return group_bbox(strokes)


def _overlap_x(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def _overlap_y(a, b) -> float:
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _gap_x(a, b) -> float:
    # positive if b starts to the right of a's end
    return b[0] - a[2]


def _min_distance(group: Sequence[np.ndarray], stroke: np.ndarray) -> float:
    """Smallest distance between `stroke` and any stroke already in `group`.

    Sampled to keep it cheap. ~0 means the strokes touch or cross.
    """

    def sample(a, k=16):
        if a.shape[0] <= k:
            return a
        idx = np.linspace(0, a.shape[0] - 1, k).astype(int)
        return a[idx]

    b = sample(stroke)
    best = np.inf
    for s in group:
        a = sample(s)
        # pairwise distances between a (Na,2) and b (Nb,2)
        diff = a[:, None, :] - b[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2)).min()
        best = min(best, float(d))
    return best


def _should_merge(group, group_box, next_stroke, next_box, median_w, median_h) -> bool:
    """Merge a stroke into the current symbol only if it belongs to that glyph.

    Rule: strokes of one symbol either physically touch/cross (e.g. '+', the
    multiplication cross, a crossed digit) or sit immediately adjacent. Purely
    stacked strokes that don't touch ('=', the division sign, a fraction bar
    over its numerator/denominator) are deliberately kept separate and
    reassembled later by the layout stage, which can see the 2D structure.

    A third case: digits like '4', '5', '7' are often drawn as a top stroke
    and a main body that don't physically touch but sit in the same column
    (significant x-overlap, small vertical gap). The previous rules let
    those split into two symbols, so a two-stroke '4' came out as '01'.
    The new rule below merges them while still rejecting the stacked cases.
    """
    touch_thr = max(4.0, 0.16 * median_h)
    if _min_distance(group, next_stroke) <= touch_thr:
        return True

    # immediately adjacent horizontally and within the same vertical band
    gap_x = _gap_x(group_box, next_box)
    oy = _overlap_y(group_box, next_box)
    gh = max(group_box[3] - group_box[1], 1e-6)
    nh = max(next_box[3] - next_box[1], 1e-6)
    oy_ratio = oy / max(1e-6, min(gh, nh))
    if 0 <= gap_x <= max(3.0, median_w * 0.08) and oy_ratio >= 0.4:
        return True

    # Two-stroke digits ('4', '5', '7'): vertically stacked but x-overlapping
    # in the same column. Require strong x-overlap, small vertical gap, and
    # a combined bounding box that looks like a normal-sized character. The
    # combined-height check protects '=' (two short bars -> small combined
    # height), and the tiny-stroke skip protects '÷' (bar + dots).
    ox = _overlap_x(group_box, next_box)
    gw = max(group_box[2] - group_box[0], 1e-6)
    nw = max(next_box[2] - next_box[0], 1e-6)
    ox_ratio = ox / max(1e-6, min(gw, nw))

    if next_box[1] > group_box[3]:
        vgap = next_box[1] - group_box[3]
    elif group_box[1] > next_box[3]:
        vgap = group_box[1] - next_box[3]
    else:
        vgap = 0.0

    combined_w = max(group_box[2], next_box[2]) - min(group_box[0], next_box[0])
    combined_h = max(group_box[3], next_box[3]) - min(group_box[1], next_box[1])
    next_is_tiny = (next_box[3] - next_box[1] < 0.35 * median_h
                    and next_box[2] - next_box[0] < 0.35 * median_w)

    if (not next_is_tiny
            and ox_ratio >= 0.40
            and abs(gap_x) <= max(6.0, median_w * 0.20)
            and vgap <= max(8.0, 0.28 * median_h)
            and combined_w <= 1.5 * median_w
            and combined_h >= 0.60 * median_h):
        return True

    return False


def segment_strokes(strokes) -> List[List[np.ndarray]]:
    """Return a list of symbol groups; each group is a list of Nx2 arrays.

    `strokes` may be raw frontend strokes or already-converted arrays.
    """
    from .strokes import clean_strokes

    arrs = clean_strokes(strokes) if (strokes and isinstance(strokes[0], (list, tuple))
                                      and (not isinstance(strokes[0], np.ndarray))) else \
        [s for s in (strokes or []) if isinstance(s, np.ndarray) and s.shape[0] >= 1]

    # Most callers pass raw frontend strokes; clean_strokes handles those.
    if not arrs:
        arrs = clean_strokes(strokes)
    if not arrs:
        return []

    items = []
    for s in arrs:
        b = _bbox_of([s])
        items.append({"stroke": s, "box": b})

    items.sort(key=lambda it: it["box"][0])  # left to right by x1

    widths = [max(4.0, it["box"][2] - it["box"][0]) for it in items]
    heights = [max(4.0, it["box"][3] - it["box"][1]) for it in items]
    median_w = float(np.median(widths))
    median_h = float(np.median(heights))

    groups: List[List[np.ndarray]] = []
    current = [items[0]["stroke"]]
    current_box = items[0]["box"]

    for it in items[1:]:
        gbox = _bbox_of(current)
        if _should_merge(current, gbox, it["stroke"], it["box"], median_w, median_h):
            current.append(it["stroke"])
        else:
            groups.append(current)
            current = [it["stroke"]]

    groups.append(current)
    return groups


def median_symbol_height(groups: Sequence[Sequence[np.ndarray]]) -> float:
    hs = []
    for g in groups:
        gm = GroupGeom(g)
        if gm.h > 0:
            hs.append(gm.h)
    return float(np.median(hs)) if hs else 1.0
