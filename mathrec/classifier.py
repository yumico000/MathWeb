import numpy as np

from .labels import CLASSES
from .enhanced_features import enhanced_feature_vector, flatten_points, paren_direction_feature

MODEL_PATH = "mathrec/model.npz"
_MODEL = None


class StrokeMLP:
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None,
                 in_dim=None, hidden=None, n_classes=None, seed=1):

        if input_dim is None:
            input_dim = in_dim
        if hidden_dim is None:
            hidden_dim = hidden
        if output_dim is None:
            output_dim = n_classes

        if input_dim is None or hidden_dim is None or output_dim is None:
            raise ValueError("StrokeMLP needs input_dim/hidden_dim/output_dim or in_dim/hidden/n_classes")

        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(0, 0.15, size=(input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = rng.normal(0, 0.15, size=(hidden_dim, output_dim)).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.std = np.ones(input_dim, dtype=np.float32)

    def fit(self, X, y, epochs=300, lr=0.03, weight_decay=1e-4, verbose=True):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n = X.shape[0]
        num_classes = self.b2.shape[0]

        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32) + 1e-6

        Xn = (X - self.mean) / self.std

        Y = np.zeros((n, num_classes), dtype=np.float32)
        Y[np.arange(n), y] = 1.0

        for epoch in range(epochs):
            H = np.tanh(Xn @ self.W1 + self.b1)
            logits = H @ self.W2 + self.b2

            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-8))
            loss += weight_decay * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))

            dlogits = (probs - Y) / n

            dW2 = H.T @ dlogits + 2 * weight_decay * self.W2
            db2 = dlogits.sum(axis=0)

            dH = dlogits @ self.W2.T
            dZ1 = dH * (1 - H * H)

            dW1 = Xn.T @ dZ1 + 2 * weight_decay * self.W1
            db1 = dZ1.sum(axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                pred = probs.argmax(axis=1)
                acc = float((pred == y).mean())
                print(f"epoch {epoch:3d}  loss {loss:.4f}  train_acc {acc:.3f}")

    def forward(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xs = (X - self.mean) / (self.std + 1e-8)

        h = np.tanh(Xs @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2

        return logits, h, Xs


    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xn = (X - self.mean) / (self.std + 1e-8)

        H = np.tanh(Xn @ self.W1 + self.b1)
        logits = H @ self.W2 + self.b2

        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def proba(self, X):
        return self.predict_proba(X)


    def save(self, path=MODEL_PATH):
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            mean=self.mean,
            std=self.std,
        )

    @classmethod
    def load(cls, path=MODEL_PATH):
        data = np.load(path, allow_pickle=True)

        input_dim = data["W1"].shape[0]
        hidden_dim = data["W1"].shape[1]
        output_dim = data["W2"].shape[1]

        model = cls(input_dim, hidden_dim, output_dim)

        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.mean = data["mean"]
        model.std = data["std"]

        return model


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def load_model(path=MODEL_PATH):
    data = np.load(path, allow_pickle=True)
    return {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
        "mean": data["mean"],
        "std": data["std"],
    }


def get_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = load_model()

    return _MODEL


def _path_length(pts):
    if pts is None or len(pts) < 2:
        return 0.0

    diffs = np.diff(pts, axis=0)
    return float(np.sqrt((diffs * diffs).sum(axis=1)).sum())


def _open_arc_ratio(group):
    pts = flatten_points(group)

    if len(pts) < 8:
        return 0.0

    length = _path_length(pts)

    if length <= 1e-6:
        return 0.0

    endpoint_dist = float(np.linalg.norm(pts[-1] - pts[0]))
    return endpoint_dist / length


def _shape_stats(group):
    pts = flatten_points(group)

    if len(pts) == 0:
        return None

    xs = pts[:, 0]
    ys = pts[:, 1]

    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())

    if width <= 1 or height <= 1:
        return None

    return {
        "width": width,
        "height": height,
        "aspect_hw": height / width,
        "direction": paren_direction_feature(group),
        "open_ratio": _open_arc_ratio(group),
    }


def parenthesis_shape_override(group):
    st = _shape_stats(group)

    if st is None:
        return None

    # Parentheses are usually one stroke, sometimes two.
    if len(group) > 2:
        return None

    # Tall and narrow.
    if st["aspect_hw"] < 1.45:
        return None

    # Important: real parentheses are open arcs.
    # Closed-loop 0, 6, 8 should NOT be forced to paren.
    if st["open_ratio"] < 0.38:
        return None

    # Clear left/right belly direction.
    if abs(st["direction"]) < 0.08:
        return None

    if st["direction"] < 0:
        return "lparen", 0.995

    return "rparen", 0.995


def neural_topk(group, top_k=5):
    model = get_model()

    feat = enhanced_feature_vector(group).astype(np.float32)

    mean = model["mean"]
    std = model["std"]

    feat = (feat - mean) / (std + 1e-8)

    h = np.tanh(feat @ model["W1"] + model["b1"])
    logits = h @ model["W2"] + model["b2"]

    probs = _softmax(logits)

    idx = np.argsort(probs)[::-1][:top_k]
    return [(CLASSES[int(i)], float(probs[int(i)])) for i in idx]


def _renormalize(top):
    total = sum(float(p) for _, p in top)

    if total <= 0:
        return top

    return [(lab, float(p) / total) for lab, p in top]


def classify_group(group, top_k=5):
    top = neural_topk(group, top_k=max(top_k, 5))

    forced = parenthesis_shape_override(group)

    if forced is not None:
        lab, conf = forced

        scores = {k: v * 0.15 for k, v in top}
        scores[lab] = max(scores.get(lab, 0.0), conf)

        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return _renormalize(top)

    st = _shape_stats(group)

    # If a closed-loop digit is predicted as parenthesis, push paren down.
    if st is not None and top and top[0][0] in {"lparen", "rparen"}:
        if st["open_ratio"] < 0.30:
            non_paren = [(lab, p) for lab, p in top if lab not in {"lparen", "rparen"}]
            paren = [(lab, p * 0.05) for lab, p in top if lab in {"lparen", "rparen"}]

            fixed = sorted(non_paren + paren, key=lambda kv: kv[1], reverse=True)[:top_k]

            if fixed:
                return _renormalize(fixed)

    return _renormalize(top[:top_k])
