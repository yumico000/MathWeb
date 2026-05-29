"""
mathrec - online (stroke-based) handwritten math recognition.

This package recognizes handwritten math from *stroke data* (the pen
trajectory captured by the canvas), not from a flattened bitmap. Working
from strokes preserves the cues - aspect ratio, curvature, stroke order -
that are destroyed when each symbol is cropped and resized to a square
image, which is what made parentheses, division bars and multi-symbol
expressions unreliable in the old image-only pipeline.

Pipeline:
    strokes -> segment -> classify (top-k) -> 2D layout -> text / LaTeX

The whole package is pure Python + numpy, so it runs with no extra
dependencies. The existing CNN can still be plugged in as an optional
shape booster (see recognize.py / cnn_adapter.py), but is never required.

Public API:
    recognize_strokes(strokes)            -> dict with text, latex, segments
    check_answer(student, correct)        -> dict with ok / equivalent
"""

from .recognize import recognize_strokes
from .checker import check_answer, is_equivalent

__all__ = ["recognize_strokes", "check_answer", "is_equivalent"]
