"""
Correctness checking with SymPy.

We never string-match answers. Instead we parse the student's input and the
expected answer into SymPy and test mathematical equivalence, so "7/3",
"14/6", "2 + 1/3" and "2.333..." are all judged on their actual value.

Accepts plain text ("(3/12)+8"), unicode operators, a leading equation
("(4/6)/3 = 2/9"), and light LaTeX (\frac, \times, \div, ^).
"""

from __future__ import annotations

import re
from typing import Any, Dict

import sympy as sp


# ----------------------------------------------------------------------------
# Input normalization
# ----------------------------------------------------------------------------
def _latex_to_plain(s: str) -> str:
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    # \frac{a}{b} -> ((a)/(b))   (repeat for nested)
    frac = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    while frac.search(s):
        s = frac.sub(r"((\1)/(\2))", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", "")
    return s


def normalize(text: str) -> str:
    if not text:
        return ""
    s = str(text).strip()
    s = _latex_to_plain(s)
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("^", "**")
    s = s.replace(" ", "")
    return s


def _to_expr(text: str):
    """Parse text into a SymPy expression (raises on failure)."""
    s = normalize(text)
    if not s:
        raise ValueError("empty input")
    return sp.simplify(sp.sympify(s, rational=True))


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def is_equivalent(a: str, b: str) -> bool:
    """True if expressions a and b are mathematically equal."""
    try:
        ea, eb = _to_expr(a), _to_expr(b)
        return bool(sp.simplify(ea - eb) == 0)
    except Exception:
        return False


def check_answer(student: str, correct: str) -> Dict[str, Any]:
    """Check a student answer against the expected value.

    `student` may be an expression or an equation 'lhs = rhs'. If it is an
    equation, both sides must be equal AND equal to `correct`.
    Returns a dict: {ok, equivalent, reason, student_value, correct_value}.
    """
    result: Dict[str, Any] = {
        "ok": True, "equivalent": False, "reason": "",
        "student_value": None, "correct_value": None,
    }
    try:
        expected = _to_expr(correct)
        result["correct_value"] = str(expected)
    except Exception as e:
        result["ok"] = False
        result["reason"] = f"Could not parse expected answer: {e}"
        return result

    raw = (student or "").strip()
    if not raw:
        result["reason"] = "No answer provided."
        return result

    try:
        if "=" in normalize(raw):
            lhs, rhs = normalize(raw).split("=", 1)
            el = sp.simplify(sp.sympify(lhs, rational=True))
            er = sp.simplify(sp.sympify(rhs, rational=True))
            eq_true = sp.simplify(el - er) == 0
            val_ok = sp.simplify(el - expected) == 0
            result["student_value"] = str(el)
            result["equivalent"] = bool(eq_true and val_ok)
            if not eq_true:
                result["reason"] = "The two sides of your equation are not equal."
            elif not val_ok:
                result["reason"] = "Your equation is true but not equal to the expected value."
        else:
            ev = _to_expr(raw)
            result["student_value"] = str(ev)
            result["equivalent"] = bool(sp.simplify(ev - expected) == 0)
            if not result["equivalent"]:
                result["reason"] = "Not equal to the expected value."
    except Exception as e:
        result["ok"] = False
        result["reason"] = f"Could not parse your answer: {e}"
    return result
