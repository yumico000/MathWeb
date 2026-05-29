"""Canonical symbol classes and their text / LaTeX renderings."""

# The class set covers the arithmetic / PEMDAS domain the app uses today.
# Add classes here (and a template in synth.py) to extend the recognizer.
CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "plus", "minus", "times", "divide", "equals",
    "lparen", "rparen", "dot", "slash",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Plain-text (Python/Sympy-friendly) rendering used to build the answer string.
LABEL_TO_TEXT = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "divide": "/",
    "equals": "=",
    "lparen": "(",
    "rparen": ")",
    "dot": ".",
    "slash": "/",
}

# LaTeX rendering for display.
LABEL_TO_LATEX = {
    "plus": "+",
    "minus": "-",
    "times": r"\times",
    "divide": r"\div",
    "equals": "=",
    "lparen": "(",
    "rparen": ")",
    "dot": ".",
    "slash": "/",
}


def label_text(label: str) -> str:
    return LABEL_TO_TEXT.get(label, label)


def label_latex(label: str) -> str:
    return LABEL_TO_LATEX.get(label, label)
