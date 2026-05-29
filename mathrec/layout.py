"""
2D layout parsing: positioned symbols -> expression text + LaTeX.

This is where the structural decisions live - deliberately, so the
classifier never has to make a call it cannot make in isolation:

  * minus vs fraction bar: a horizontal stroke is a fraction bar only if
    there is content both ABOVE and BELOW it across its width; otherwise
    it is a minus. The classifier just reports "minus"; we decide here.
  * exponents: a small symbol raised above the running baseline becomes a
    superscript rather than a same-line token.
  * parentheses are emitted as grouping characters.

Output is both a Python/Sympy-friendly text string and a LaTeX string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .labels import label_latex, label_text


@dataclass
class Sym:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    text: str = ""
    latex: str = ""

    def __post_init__(self):
        if not self.text:
            self.text = label_text(self.label)
        if not self.latex:
            self.latex = label_latex(self.label)

    @property
    def cx(self): return (self.x1 + self.x2) / 2.0
    @property
    def cy(self): return (self.y1 + self.y2) / 2.0
    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1


def _median_height(syms: List[Sym]) -> float:
    hs = [s.h for s in syms if s.h > 0]
    return float(np.median(hs)) if hs else 1.0


def _median_width(syms: List[Sym]) -> float:
    ws = [s.w for s in syms if s.w > 0]
    return float(np.median(ws)) if ws else 1.0


def _is_fraction_bar(bar: Sym, others: List[Sym], med_w: float) -> bool:
    """A 'minus' is a fraction bar iff content sits above AND below its span."""
    if bar.label not in ("minus",):
        return False
    if bar.w < 1.3 * med_w:  # too short to span a fraction
        return False
    pad = 0.15 * bar.w
    lo, hi = bar.x1 - pad, bar.x2 + pad
    above = below = False
    for s in others:
        if s is bar:
            continue
        if lo <= s.cx <= hi:
            if s.cy < bar.cy - 0.2 * max(bar.h, 1.0):
                above = True
            elif s.cy > bar.cy + 0.2 * max(bar.h, 1.0):
                below = True
    return above and below


def parse(syms: List[Sym]) -> dict:
    """Parse a flat list of positioned symbols into text + latex."""
    syms = [s for s in syms if s is not None]
    if not syms:
        return {"text": "", "latex": ""}

    med_w = _median_width(syms)

    # ---- 1. resolve fraction bars (recursively) ----------------------------
    bars = [s for s in syms if _is_fraction_bar(s, syms, med_w)]
    # process widest bars first so nested fractions resolve inner-first later
    bars.sort(key=lambda b: b.w, reverse=True)

    consumed = set()
    tokens: List[Sym] = []

    for bar in bars:
        if id(bar) in consumed:
            continue
        pad = 0.15 * bar.w
        lo, hi = bar.x1 - pad, bar.x2 + pad
        num, den = [], []
        for s in syms:
            if s is bar or id(s) in consumed:
                continue
            if lo <= s.cx <= hi:
                if s.cy < bar.cy:
                    num.append(s)
                elif s.cy > bar.cy:
                    den.append(s)
        if not num or not den:
            continue
        for s in num + den:
            consumed.add(id(s))
        consumed.add(id(bar))

        num_res = parse(num)
        den_res = parse(den)
        frac = Sym(
            label="_frac",
            x1=bar.x1, y1=min(s.y1 for s in num),
            x2=bar.x2, y2=max(s.y2 for s in den),
            text=f"(({num_res['text']})/({den_res['text']}))",
            latex=rf"\frac{{{num_res['latex']}}}{{{den_res['latex']}}}",
        )
        tokens.append(frac)

    for s in syms:
        if id(s) not in consumed:
            tokens.append(s)

    # ---- 2. linear pass left to right, with exponent detection -------------
    tokens.sort(key=lambda s: s.cx)
    base_h = _median_height([t for t in tokens if t.label not in ("dot",)]) or 1.0

    text_parts: List[str] = []
    latex_parts: List[str] = []
    prev = None
    for t in tokens:
        is_exponent = False
        if prev is not None and t.label in list("0123456789"):
            # raised and smaller than the base it follows
            raised = t.cy < prev.cy - 0.18 * max(prev.h, base_h)
            smaller = t.h < 0.82 * max(prev.h, base_h)
            after_basey = prev.label in list("0123456789") + ["rparen", "_frac"]
            if raised and smaller and after_basey:
                is_exponent = True

        if is_exponent:
            text_parts.append("**" + t.text)
            latex_parts.append("^{" + t.latex + "}")
        else:
            text_parts.append(t.text)
            latex_parts.append(t.latex)
        prev = t

    return {"text": "".join(text_parts), "latex": "".join(latex_parts)}
