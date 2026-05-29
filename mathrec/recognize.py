"""
Recognizer orchestrator.

    strokes -> segment -> classify each group (top-k) -> assemble stacked
    composites -> 2D layout -> text/LaTeX

Design points from our discussion:

  1. Geometric rules (computed from the trajectory: aspect ratio and
     belly-vs-endpoints x) disambiguate the symbols a bitmap classifier
     gets wrong - parentheses direction, dots, the horizontal bar. They
     are reliable, so they override the learned classifier when confident.

  2. We keep top-k candidates per symbol and let a light grammar pass pick a
     globally consistent reading (e.g. balanced parentheses) instead of
     committing to top-1.

  3. Stacked multi-stroke symbols ('=', the division sign) are reassembled
     here, where the 2D arrangement is visible; minus-vs-fraction-bar and
     exponents are decided in the layout stage.

The old image CNN can be added as one more vote via cnn_adapter (optional).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from . import strokes as S
from .classifier import classify_group
from .labels import label_latex, label_text
from .layout import Sym, parse
from .segment import segment_strokes


# ----------------------------------------------------------------------------
# Geometric helpers (trajectory based)
# ----------------------------------------------------------------------------
def _paren_direction(group: List[np.ndarray]):
    """Decide '(' vs ')' from the stroke shape, independent of draw direction.

    A parenthesis has two endpoints (its tips) and a belly that bulges to one
    side. For '(' the belly sits to the LEFT of the endpoints; for ')' it sits
    to the RIGHT. Comparing the belly's x to the endpoints' x is invariant to
    which way the arc was drawn and to the y-axis orientation - unlike the
    signed curvature, whose sign flips with drawing direction. Returns
    'lparen', 'rparen', or None if inconclusive.
    """
    pts = S.group_to_points(group)
    if pts.shape[0] < 5:
        return None
    x1, _, x2, _ = S.bbox(pts)
    w = max(x2 - x1, 1e-6)

    order = np.argsort(pts[:, 1])          # sort by y to find the two tips
    p = pts[order]
    n = len(p)
    q = max(1, n // 4)
    top_x = p[:q, 0].mean()
    bot_x = p[-q:, 0].mean()
    end_avg_x = (top_x + bot_x) / 2.0
    mid = p[n // 3: max(n // 3 + 1, 2 * n // 3)]
    mid_x = mid[:, 0].mean()

    if abs(mid_x - end_avg_x) < 0.12 * w:  # too straight to tell
        return None
    return "lparen" if mid_x < end_avg_x else "rparen"


def _geometric_vote(group: List[np.ndarray], med_h: float, med_w: float):
    """Return (label, confidence) or None from pure geometry."""
    pts = S.group_to_points(group)
    if pts.shape[0] == 0:
        return None
    x1, y1, x2, y2 = S.bbox(pts)
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    aspect = w / h
    res = S.resample(pts, 24)
    turning = S.total_turning(res)
    nst = len(group)

    # dot / decimal point: tiny in both dimensions
    if w < 0.45 * med_w and h < 0.45 * med_h and S.arc_length(pts) < 0.9 * med_w:
        return ("dot", 0.9)

    # horizontal bar -> minus (layout may promote it to a fraction bar)
    if aspect > 3.2 and turning < 1.2 and nst == 1:
        return ("minus", 0.85)

    # Plain vertical "1" (no flag): tall, very thin, nearly straight,
    # trajectory dominated by vertical motion. Many writers draw '1' as a
    # bare vertical bar, but the MathWriting templates flag the top, so the
    # learned classifier mis-routes these to 'slash' or '('. This geometric
    # check rescues them; it does NOT fire on a flagged '1' (turning higher)
    # or on a paren (more curved) or on a slash (more diagonal).
    if nst == 1 and aspect < 0.35 and turning < 1.0:
        d = res[-1] - res[0]
        if abs(d[1]) > 2.5 * (abs(d[0]) + 1e-6):
            return ("1", 0.8)

    # NOTE: we deliberately do NOT detect parentheses geometrically here -
    # narrow curvy digits (1, 3, 9) look similar and would be misread. The
    # learned classifier separates parens from digits; we only correct the
    # left/right *direction* of a paren (in _classify_one) via shape.

    # Straight diagonal single stroke -> slash.
    # Aspect window tightened to 0.6..1.8 so narrow parens (aspect ~0.3-0.5)
    # cannot leak in; the 'lparen -> slash' confusion was caused by this.
    if nst == 1 and turning < 0.8 and 0.6 < aspect < 1.8:
        d = res[-1] - res[0]
        if d[0] < -0.35 * w and d[1] > 0.35 * h:
            return ("slash", 0.7)

    return None


def _call_classifier(group, top_k=5):
    """Adapter for both classifier signatures.

    Old API: classify_group(group, top_k=5) -> [(label, prob), ...]
    New API: classify_group(group)          -> (best_label, best_conf,
                                                 [(label, prob), ...])
    Returns a normalized list of (label, prob) pairs either way.
    """
    try:
        raw = classify_group(group, top_k=top_k)
    except TypeError:
        raw = classify_group(group)
    if isinstance(raw, tuple) and len(raw) >= 3 and isinstance(raw[2], (list, tuple)):
        return [(str(l), float(p)) for l, p in list(raw[2])[:top_k]]
    if isinstance(raw, tuple) and len(raw) == 2 and not isinstance(raw[0], (list, tuple)):
        return [(str(raw[0]), float(raw[1]))]
    return [(str(l), float(p)) for l, p in list(raw)[:top_k]]


def _classify_one(group, med_h, med_w, top_k=5):
    """Combine learned classifier + geometric rules; return top-k list."""
    learned = _call_classifier(group, top_k=top_k)
    scores = {lab: p for lab, p in learned}

    geo = _geometric_vote(group, med_h, med_w)
    if geo is not None:
        glab, gconf = geo
        scores[glab] = max(scores.get(glab, 0.0), gconf) + 0.5 * gconf

    # optional CNN vote (only if torch + model + PIL are present)
    try:
        from .cnn_adapter import cnn_vote
        for lab, p in (cnn_vote(group) or []):
            scores[lab] = scores.get(lab, 0.0) + 0.4 * p
    except Exception:
        pass

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    # Parenthesis direction is decided by stroke shape (belly vs endpoints),
    # which is far more reliable than the learned classifier and does NOT
    # depend on which way the arc was drawn.
    if ranked and ranked[0][0] in ("lparen", "rparen"):
        correct = _paren_direction(group)
        if correct is not None:
            ranked = [(correct, ranked[0][1])] + [r for r in ranked if r[0] != correct]

    tot = sum(v for _, v in ranked) or 1.0
    return [(lab, v / tot) for lab, v in ranked]


# ----------------------------------------------------------------------------
# Grammar pass: keep top-k, choose a consistent reading (balance parens)
# ----------------------------------------------------------------------------
def _resolve_candidates(per_symbol_topk: List[List]):
    chosen = [tk[0][0] for tk in per_symbol_topk]

    def balanced(labels):
        depth = 0
        for l in labels:
            if l == "lparen":
                depth += 1
            elif l == "rparen":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    if balanced(chosen):
        return chosen
    for i, tk in enumerate(per_symbol_topk):
        opts = {l for l, _ in tk}
        if chosen[i] in ("lparen", "rparen") and {"lparen", "rparen"} & opts:
            trial = list(chosen)
            trial[i] = "rparen" if chosen[i] == "lparen" else "lparen"
            if balanced(trial):
                return trial
    return chosen


# ----------------------------------------------------------------------------
# Assemble stacked composites ('=' and the division sign) from split strokes
# ----------------------------------------------------------------------------
def _x_overlap_ratio(a: Sym, b: Sym) -> float:
    ov = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    return ov / max(1e-6, min(a.w, b.w))


def _assemble(syms: List[Sym], med_h: float, med_w: float) -> List[Sym]:
    used = [False] * len(syms)
    out: List[Sym] = []

    # division sign: a bar with a dot above and a dot below.
    # The bar width threshold was 1.6 * med_w but real ÷ bars are often longer;
    # at 1.6x we missed the assembly and the layout parser then mistook the
    # arrangement for a fraction (.)/(.). Loosened to 2.4x.
    for i, m in enumerate(syms):
        if used[i] or m.label != "minus" or m.w > 2.4 * med_w:
            continue
        above = below = -1
        for j, d in enumerate(syms):
            if used[j] or j == i or d.label != "dot":
                continue
            if m.x1 - 0.5 * m.w <= d.cx <= m.x2 + 0.5 * m.w:
                if d.cy < m.cy:
                    above = j
                elif d.cy > m.cy:
                    below = j
        if above >= 0 and below >= 0:
            used[i] = used[above] = used[below] = True
            ys = [syms[above].y1, m.y1, syms[below].y1]
            ye = [syms[above].y2, m.y2, syms[below].y2]
            div = Sym(label="divide", x1=m.x1, y1=min(ys), x2=m.x2, y2=max(ye))
            div.conf = getattr(m, "conf", 0.7)
            div.topk = [("divide", div.conf)]
            out.append(div)

    # equals: two similar short bars stacked close together
    minus_idx = [i for i, s in enumerate(syms)
                 if s.label == "minus" and not used[i]]
    for a in range(len(minus_idx)):
        ia = minus_idx[a]
        if used[ia]:
            continue
        for b in range(a + 1, len(minus_idx)):
            ib = minus_idx[b]
            if used[ib]:
                continue
            A, B = syms[ia], syms[ib]
            wide = max(A.w, B.w) > 2.0 * med_w  # likely a fraction bar, skip
            similar = 0.4 <= (A.w / max(B.w, 1e-6)) <= 2.5
            gap = abs(A.cy - B.cy)
            if (not wide and similar and _x_overlap_ratio(A, B) > 0.45
                    and 0.08 * med_h < gap < 0.8 * med_h):
                used[ia] = used[ib] = True
                eq = Sym(label="equals",
                         x1=min(A.x1, B.x1), y1=min(A.y1, B.y1),
                         x2=max(A.x2, B.x2), y2=max(A.y2, B.y2))
                eq.conf = min(getattr(A, "conf", 0.8), getattr(B, "conf", 0.8))
                eq.topk = [("equals", eq.conf)]
                out.append(eq)
                break

    for i, s in enumerate(syms):
        if not used[i]:
            out.append(s)
    out.sort(key=lambda s: s.cx)
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def recognize_strokes(strokes) -> Dict[str, Any]:
    """Recognize handwritten math from frontend stroke data.

    Returns {ok, text, latex, segments:[...]} matching the app's contract.
    """
    groups = segment_strokes(strokes)
    if not groups:
        return {"ok": False, "error": "No strokes found. Draw something and try again."}

    geoms = [S.GroupGeom(g) for g in groups]
    med_h = float(np.median([gm.h for gm in geoms if gm.h > 0]) or 1.0)
    med_w = float(np.median([gm.w for gm in geoms if gm.w > 0]) or 1.0)

    per_symbol_topk = [_classify_one(g, med_h, med_w) for g in groups]
    chosen = _resolve_candidates(per_symbol_topk)

    syms: List[Sym] = []
    for g, gm, topk, lab in zip(groups, geoms, per_symbol_topk, chosen):
        s = Sym(label=lab, x1=gm.x1, y1=gm.y1, x2=gm.x2, y2=gm.y2)
        s.conf = dict(topk).get(lab, topk[0][1])
        s.topk = topk
        syms.append(s)

    syms = _assemble(syms, med_h, med_w)
    layout = parse(syms)

    segments = []
    for i, s in enumerate(syms):
        topk = getattr(s, "topk", [(s.label, getattr(s, "conf", 1.0))])
        segments.append({
            "index": i + 1,
            "box": [int(s.x1), int(s.y1), int(s.x2), int(s.y2)],
            "label": s.label,
            "text": label_text(s.label),
            "confidence": float(getattr(s, "conf", topk[0][1])),
            "top_predictions": [
                {"label": l, "text": label_text(l), "confidence": float(p)}
                for l, p in topk
            ],
        })

    return {
        "ok": True,
        "text": layout["text"],
        "latex": layout["latex"],
        "segments": segments,
    }
