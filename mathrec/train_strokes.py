"""
Train the stroke classifier.

Default: trains on synthetic data (no download needed) and writes model.npz.

    python -m mathrec.train_strokes

To train on REAL data, build (X, y) yourself - X is an array of
strokes.feature_vector(group) rows, y is class indices from labels.CLASSES -
and call train(X, y). Collecting real strokes from the canvas, or parsing
CROHME InkML traces, will dramatically outperform the synthetic baseline on
genuine student handwriting.
"""

from __future__ import annotations

import numpy as np

from .classifier import StrokeMLP, MODEL_PATH
from .labels import CLASSES


def train(X, y, hidden=96, epochs=120, batch=128, lr=2e-3, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    n_classes = len(CLASSES)

    model = StrokeMLP(in_dim=d, hidden=hidden, n_classes=n_classes)
    model.mean = X.mean(axis=0)
    model.std = X.std(axis=0)

    # He-ish init for the ReLU layer
    model.W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
    model.b1 = np.zeros(hidden)
    model.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, n_classes))
    model.b2 = np.zeros(n_classes)

    Y = np.eye(n_classes)[y]

    # Adam state
    params = {"W1": model.W1, "b1": model.b1, "W2": model.W2, "b2": model.b2}
    mom = {k: np.zeros_like(v) for k, v in params.items()}
    vel = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0

    idx = np.arange(n)
    for ep in range(epochs):
        rng.shuffle(idx)
        total_loss = 0.0
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            Xb, Yb = X[bi], Y[bi]
            logits, h, Xs = model.forward(Xb)

            # softmax + cross entropy
            z = logits - logits.max(axis=1, keepdims=True)
            e = np.exp(z)
            probs = e / e.sum(axis=1, keepdims=True)
            m = Xb.shape[0]
            loss = -np.log((probs * Yb).sum(axis=1) + 1e-9).mean()
            total_loss += loss * m

            # gradients
            dlogits = (probs - Yb) / m
            gW2 = h.T @ dlogits
            gb2 = dlogits.sum(axis=0)
            dh = dlogits @ model.W2.T
            dh[h <= 0] = 0.0
            gW1 = Xs.T @ dh
            gb1 = dh.sum(axis=0)

            grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
            step += 1
            for k in params:
                mom[k] = b1 * mom[k] + (1 - b1) * grads[k]
                vel[k] = b2 * vel[k] + (1 - b2) * (grads[k] ** 2)
                mhat = mom[k] / (1 - b1 ** step)
                vhat = vel[k] / (1 - b2 ** step)
                params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
            model.W1, model.b1, model.W2, model.b2 = (
                params["W1"], params["b1"], params["W2"], params["b2"])

        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            preds = model.proba(X).argmax(axis=1)
            acc = (preds == y).mean()
            print(f"epoch {ep:3d}  loss {total_loss / n:.4f}  train_acc {acc:.3f}")

    model.trained = True
    return model


def main():
    from .synth import make_dataset

    print("Generating synthetic stroke dataset...")
    X, y = make_dataset(per_class=500, seed=1)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features, {len(CLASSES)} classes")

    # simple train/val split
    rng = np.random.default_rng(2)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    cut = int(0.85 * len(y))
    Xtr, ytr, Xva, yva = X[:cut], y[:cut], X[cut:], y[cut:]

    model = train(Xtr, ytr, epochs=140)
    val_acc = (model.proba(Xva).argmax(axis=1) == yva).mean()
    print(f"validation accuracy (synthetic): {val_acc:.3f}")

    model.save(MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
