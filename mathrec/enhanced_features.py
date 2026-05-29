import numpy as np
from mathrec.strokes import feature_vector


def _is_point(obj):
    if isinstance(obj, dict):
        return "x" in obj and "y" in obj

    if isinstance(obj, (list, tuple, np.ndarray)):
        return len(obj) >= 2 and isinstance(obj[0], (int, float, np.number))

    return False


def _point_xy(p):
    if isinstance(p, dict):
        return float(p.get("x", 0)), float(p.get("y", 0))

    return float(p[0]), float(p[1])


def normalize_strokes(strokes):
    """
    Convert browser stroke format into the format mathrec/strokes.py expects:
        [np.array([[x,y], [x,y], ...]), np.array(...)]
    """
    if strokes is None:
        return []

    # Case: one stroke given directly as list of points
    if isinstance(strokes, (list, tuple)) and len(strokes) > 0 and _is_point(strokes[0]):
        one = []
        for p in strokes:
            try:
                x, y = _point_xy(p)
                one.append([x, y])
            except Exception:
                pass

        if not one:
            return []

        return [np.asarray(one, dtype=np.float32)]

    out = []

    for stroke in strokes:
        if _is_point(stroke):
            try:
                x, y = _point_xy(stroke)
                out.append(np.asarray([[x, y]], dtype=np.float32))
            except Exception:
                pass
            continue

        one = []

        for p in stroke:
            try:
                x, y = _point_xy(p)
                one.append([x, y])
            except Exception:
                pass

        if one:
            out.append(np.asarray(one, dtype=np.float32))

    return out


def flatten_points(strokes):
    norm = normalize_strokes(strokes)
    pts = []

    for stroke in norm:
        if len(stroke) > 0:
            pts.append(stroke)

    if not pts:
        return np.zeros((0, 2), dtype=np.float32)

    return np.vstack(pts).astype(np.float32)


def paren_direction_feature(strokes):
    pts = flatten_points(strokes)

    if len(pts) < 5:
        return 0.0

    xs = pts[:, 0]
    ys = pts[:, 1]

    y_min = float(ys.min())
    y_max = float(ys.max())
    height = y_max - y_min

    if height <= 1:
        return 0.0

    top = pts[ys <= y_min + 0.30 * height]
    mid = pts[(ys >= y_min + 0.35 * height) & (ys <= y_min + 0.65 * height)]
    bottom = pts[ys >= y_min + 0.70 * height]

    if len(top) == 0 or len(mid) == 0 or len(bottom) == 0:
        return 0.0

    end_x = (float(top[:, 0].mean()) + float(bottom[:, 0].mean())) / 2.0
    mid_x = float(mid[:, 0].mean())

    width = float(xs.max() - xs.min())

    if width <= 1:
        return 0.0

    return (mid_x - end_x) / width


def enhanced_feature_vector(strokes):
    norm = normalize_strokes(strokes)

    base = np.asarray(feature_vector(norm), dtype=np.float32)
    direction = np.array([paren_direction_feature(norm)], dtype=np.float32)

    return np.concatenate([base, direction])
