# mathrec — stroke-based handwritten math recognition

This package replaces the old "render each symbol to an image and classify
it in isolation" pipeline. That approach is what made **parentheses,
division, and multi-symbol expressions** unreliable: resizing each symbol to
a square bitmap throws away the two cues that identify those symbols —
**aspect ratio** and **curvature/stroke direction** — and classifying each
crop alone gives the model no way to tell a minus from a fraction bar.

`mathrec` works from the **stroke data the canvas already captures**
(`{x, y, t}` per point) and is pure Python + numpy + sympy, so it runs with
no extra dependencies and no paid API.

## Pipeline

```
strokes → segment → classify each group (top-k) → assemble stacked
          composites (= ÷) → 2D layout → text + LaTeX → SymPy check
```

Key design decisions (each fixes one of the original failure modes):

- **Trajectory + geometry features** (`strokes.py`) — the classifier sees the
  resampled pen path, a direction histogram, aspect ratio and signed
  curvature, not a bitmap. This is what separates `(` `)` `1` `/` `-`.
- **Touch-based segmentation** (`segment.py`) — strokes merge into one symbol
  only if they physically touch/cross (`+`, `×`, crossed digits). Stacked
  strokes that don't touch (`=`, `÷`, a fraction bar over its
  numerator/denominator) stay separate and are rebuilt later.
- **Curvature decides paren direction** (`recognize.py`) — left-vs-right `(` /
  `)` is set from the sign of the trajectory's curvature, which is far more
  reliable than a bitmap classifier.
- **Layout defers context-dependent symbols** (`layout.py`) — a horizontal
  stroke is a **fraction bar** only if there is content both above *and*
  below it; otherwise it is a **minus**. Exponents are detected as raised,
  smaller symbols. The classifier never has to make these calls alone.
- **Top-k + grammar** — candidates are kept per symbol and a light pass picks
  a balanced-parenthesis reading instead of committing to top-1.

## Files

| file | purpose |
|------|---------|
| `strokes.py` | preprocessing + feature vector |
| `segment.py` | group strokes into symbol candidates |
| `synth.py` | parametric stroke templates + synthetic data generator |
| `classifier.py` | pure-numpy MLP over features (loads `model.npz`) |
| `train_strokes.py` | train the classifier (synthetic by default) |
| `layout.py` | positioned symbols → text + LaTeX (fractions, exponents) |
| `recognize.py` | orchestrator; public `recognize_strokes(strokes)` |
| `checker.py` | SymPy answer checking (`check_answer`, `is_equivalent`) |
| `cnn_adapter.py` | optional bridge to the old image CNN (used if torch present) |
| `model.npz` | trained baseline weights (regenerate with `train_strokes`) |

## Run

The Flask app (`app.py`) already calls this engine through
`handwriting_recognizer.py`. Just run the app as before:

```bash
python app.py
```

Tests and benchmark:

```bash
python tests/test_pipeline.py   # end-to-end correctness (expressions + checker)
python tests/benchmark.py       # randomized robustness accuracy report
```

## Train

**Baseline (no download):**

```bash
python -m mathrec.train_strokes
```

This generates synthetic strokes and writes `mathrec/model.npz`. It is enough
to make the whole pipeline run, and the benchmark scores ~100% on synthetic
data — but that number reflects pipeline correctness, **not** accuracy on real
student handwriting, because train and test come from the same templates.

**Production accuracy — train on real strokes.** Build `(X, y)`:

```python
import numpy as np
from mathrec.strokes import feature_vector
from mathrec.labels import CLASS_TO_IDX
from mathrec.train_strokes import train

X = np.array([feature_vector(group) for group in groups])  # groups = list of strokes
y = np.array([CLASS_TO_IDX[label] for label in labels])
model = train(X, y, epochs=200)
model.save()   # writes mathrec/model.npz
```

Two good sources of real `groups`:
1. **Log real canvas strokes** from your app (each labeled by the known
   problem) — this matches your users' handwriting exactly and improves fast.
2. **CROHME** (the standard handwritten-math benchmark) ships *online* stroke
   traces in InkML, which map directly onto this feature pipeline. Parse each
   trace's points into an `Nx2` array per stroke and feed `feature_vector`.

To add a new symbol: add it to `CLASSES` in `labels.py`, add a template in
`synth.py` (and/or supply real samples), then retrain.

## Using the existing CNN as a booster

If `torch`, `Pillow`, and a trained `symbol_model.pth` are present,
`cnn_adapter.cnn_vote()` automatically contributes the CNN's prediction as an
extra vote per symbol. Nothing to configure; if those aren't installed it is
silently skipped and the stroke engine runs alone.

## Recommendation: review the context "auto-correction"

`app.py`'s `/api/recognize` route runs `choose_best_context_match`, which
snaps the recognized text toward the expected answer / problem expression
(Levenshtein). For an assessment tool this can mask wrong answers — a student
who writes the wrong thing may have it "corrected" to the right answer before
checking. Consider disabling it, or use it only to fix obvious OCR noise (very
small edit distance) and never across an `=` boundary.

## Honest limitations / next steps

- The shipped model is synthetic-trained; collect/label real strokes for real
  accuracy. The architecture is the durable part — retraining is cheap.
- Stacked-fraction handling covers one bar with a numerator/denominator row;
  deeply nested fractions need recursion tuning.
- For a higher ceiling later, the same `(X, y)` data can train an end-to-end
  encoder-decoder (TAP/WAP-style) that outputs LaTeX directly.
