"""
Optional bridge to the original image-based CNN (symbol_predictor.py).

If torch, Pillow and the trained symbol_model.pth are present, cnn_vote()
renders a symbol group to an image, runs the CNN, and returns its top
predictions mapped onto mathrec's class names. The orchestrator blends this
in as one extra vote.

Everything is imported lazily and wrapped by the caller in try/except, so on
a machine without torch/Pillow (or without a trained model) this module
simply contributes nothing - the stroke engine runs unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Map the CNN's class labels onto mathrec's label set.
_CNN_TO_MATHREC = {
    "left_paren": "lparen",
    "right_paren": "rparen",
    "plus": "plus", "minus": "minus", "times": "times",
    "divide": "divide", "equals": "equals",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
}


def _group_to_image(group: List[np.ndarray], pad: int = 24, scale: int = 3):
    from PIL import Image, ImageDraw  # lazy

    pts = np.vstack([s for s in group if s.shape[0] > 0])
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    w = max(96, int((x2 - x1 + 2 * pad) * scale))
    h = max(96, int((y2 - y1 + 2 * pad) * scale))
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    for s in group:
        line = [((float(px) - x1 + pad) * scale, (float(py) - y1 + pad) * scale)
                for px, py in s]
        if len(line) >= 2:
            draw.line(line, fill="black", width=max(6, 3 * scale), joint="curve")
        elif len(line) == 1:
            cx, cy = line[0]
            r = 3 * scale
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="black")
    return img


def cnn_vote(group: List[np.ndarray]) -> Optional[List[Tuple[str, float]]]:
    """Return [(mathrec_label, prob), ...] from the CNN, or None if unavailable."""
    from symbol_predictor import predict_symbol_from_pil  # raises if torch missing

    img = _group_to_image(group)
    pred = predict_symbol_from_pil(img)
    out = []
    for p in pred.get("top_predictions", []):
        lab = _CNN_TO_MATHREC.get(str(p.get("label", "")))
        if lab:
            out.append((lab, float(p.get("confidence", 0.0))))
    return out or None
