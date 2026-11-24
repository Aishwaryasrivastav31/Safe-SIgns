"""A small convolutional network written from scratch in NumPy (NHWC layout).

conv3x3(32) -> ReLU -> maxpool2 -> conv3x3(64) -> ReLU -> maxpool2 -> dense(128) -> ReLU -> dense(43)

Forward, backward (including the gradient with respect to the input image, which the
attacks need), Adam and temperature scaling are all implemented here and checked against
numerical gradients in the tests.
"""

from __future__ import annotations

import numpy as np


def _conv_forward(x, w, b):
    """x (N,H,W,C), w (3,3,C,O), same padding."""
    N, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.zeros((N, H, W, w.shape[3]), dtype=x.dtype)
    for dy in range(3):
        for dx in range(3):
            out += xp[:, dy:dy + H, dx:dx + W, :] @ w[dy, dx]
    return out + b, xp


def _conv_backward(dout, xp, w):
    N, H, W, O = dout.shape
    dxp = np.zeros_like(xp)
    dw = np.zeros_like(w)
    flat = dout.reshape(-1, O)
    for dy in range(3):
        for dx in range(3):
            sl = xp[:, dy:dy + H, dx:dx + W, :]
            dw[dy, dx] = sl.reshape(-1, sl.shape[-1]).T @ flat
            dxp[:, dy:dy + H, dx:dx + W, :] += dout @ w[dy, dx].T
    return dxp[:, 1:-1, 1:-1, :], dw, dout.sum(axis=(0, 1, 2))


def _pool_forward(x):
    N, H, W, C = x.shape
    r = x.reshape(N, H // 2, 2, W // 2, 2, C)
    out = r.max(axis=(2, 4))
    mask = r == out[:, :, None, :, None, :]
    return out, mask


def _pool_backward(dout, mask):
    d = mask * dout[:, :, None, :, None, :]
    d = d / np.maximum(mask.sum(axis=(2, 4), keepdims=True), 1)  # split ties evenly
    N, h, _, w, _, C = d.shape
    return d.reshape(N, h * 2, w * 2, C)


class CNN:
    def __init__(self, n_classes: int = 43, size: int = 32, seed: int = 0, c1: int = 32, c2: int = 64, hidden: int = 128):
        rng = np.random.default_rng(seed)
        f = (size // 4) ** 2 * c2

        def he(shape, fan_in):
            return (rng.standard_normal(shape) * np.sqrt(2 / fan_in)).astype(np.float32)

        self.p = {"w1": he((3, 3, 3, c1), 27), "b1": np.zeros(c1, np.float32),
                  "w2": he((3, 3, c1, c2), 9 * c1), "b2": np.zeros(c2, np.float32),
                  "w3": he((f, hidden), f), "b3": np.zeros(hidden, np.float32),
                  "w4": he((hidden, n_classes), hidden), "b4": np.zeros(n_classes, np.float32)}
        self.temperature = 1.0
        self.mean = np.zeros(3, np.float32)
        self.std = np.ones(3, np.float32)

    # --- forward / backward -------------------------------------------------------------
    def _forward(self, x):
        p = self.p
        z = (x - self.mean) / self.std
        a1, xp1 = _conv_forward(z, p["w1"], p["b1"]); r1 = np.maximum(a1, 0)
        q1, m1 = _pool_forward(r1)
        a2, xp2 = _conv_forward(q1, p["w2"], p["b2"]); r2 = np.maximum(a2, 0)
        q2, m2 = _pool_forward(r2)
        f = q2.reshape(len(x), -1)
        a3 = f @ p["w3"] + p["b3"]; r3 = np.maximum(a3, 0)
        logits = r3 @ p["w4"] + p["b4"]
        cache = (xp1, a1, m1, xp2, a2, m2, q2.shape, f, a3, r3)
        return logits, cache

    def _backward(self, dlogits, cache, need_input=False):
        p = self.p
        xp1, a1, m1, xp2, a2, m2, q2shape, f, a3, r3 = cache
        g = {"w4": r3.T @ dlogits, "b4": dlogits.sum(0)}
        dr3 = dlogits @ p["w4"].T
        da3 = dr3 * (a3 > 0)
        g["w3"], g["b3"] = f.T @ da3, da3.sum(0)
        dq2 = (da3 @ p["w3"].T).reshape(q2shape)
        da2 = _pool_backward(dq2, m2) * (a2 > 0)
        dq1, g["w2"], g["b2"] = _conv_backward(da2, xp2, p["w2"])
        da1 = _pool_backward(dq1, m1) * (a1 > 0)
        dz, g["w1"], g["b1"] = _conv_backward(da1, xp1, p["w1"])
        dx = dz / self.std if need_input else None
        return g, dx

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def logits(self, x, batch=512):
        return np.concatenate([self._forward(x[i:i + batch])[0] for i in range(0, len(x), batch)])

    def predict_proba(self, x, batch=512):
        return self._softmax(self.logits(x, batch) / self.temperature)

    def loss_and_input_grad(self, x, target, targeted=False):
        """Cross-entropy gradient w.r.t. the input (untargeted: raise the true-class loss;
        targeted: lower the target-class loss)."""
        logits, cache = self._forward(x)
        prob = self._softmax(logits)
        onehot = np.eye(prob.shape[1], dtype=np.float32)[target]
        dlogits = (prob - onehot) / len(x)
        _, dx = self._backward(dlogits, cache, need_input=True)
        return -dx if targeted else dx

    # --- training ---------------------------------------------------------------------
    def fit(self, x, y, x_val, y_val, epochs=8, batch=128, lr=1e-3, weight_decay=1e-4, seed=0, augment=True, log=print,
            adv_eps: float = 0.0, adv_steps: int = 3, adv_frac: float = 0.5):
        """Train with Adam. With adv_eps > 0 this is PGD adversarial training (Madry et al.):
        in every batch, a share adv_frac of the images is replaced by a PGD attack against the
        current weights, so the network learns on its own worst cases."""
        rng = np.random.default_rng(seed)
        self.mean = x.reshape(-1, 3).mean(0).astype(np.float32)
        self.std = (x.reshape(-1, 3).std(0) + 1e-6).astype(np.float32)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(val) for k, val in self.p.items()}
        t = 0
        for ep in range(epochs):
            order = rng.permutation(len(x))
            losses = []
            for i in range(0, len(x), batch):
                idx = order[i:i + batch]
                xb, yb = x[idx], y[idx]
                if augment:
                    xb = augment_batch(xb, rng)
                if adv_eps > 0 and t > 0:
                    k = int(len(xb) * adv_frac)
                    xb = xb.copy()
                    xa = np.clip(xb[:k] + rng.uniform(-adv_eps, adv_eps, xb[:k].shape).astype(np.float32), 0, 1)
                    for _ in range(adv_steps):
                        g = self.loss_and_input_grad(xa, yb[:k])
                        xa = np.clip(np.clip(xa + (adv_eps / 2) * np.sign(g), xb[:k] - adv_eps, xb[:k] + adv_eps), 0, 1)
                    xb[:k] = xa.astype(np.float32)
                logits, cache = self._forward(xb)
                prob = self._softmax(logits)
                losses.append(float(-np.log(prob[np.arange(len(yb)), yb] + 1e-9).mean()))
                prob[np.arange(len(yb)), yb] -= 1
                g, _ = self._backward(prob / len(yb), cache)
                t += 1
                for k in self.p:
                    if k.startswith("w"):
                        g[k] = g[k] + weight_decay * self.p[k]
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                    self.p[k] -= (lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)).astype(np.float32)
            acc = float((self.logits(x_val).argmax(1) == y_val).mean())
            log(f"  epoch {ep + 1}/{epochs}: train loss {np.mean(losses):.3f}, validation accuracy {acc:.3f}")
        return self

    def calibrate_temperature(self, x_val, y_val):
        """Temperature scaling: one scalar chosen to minimise validation negative log-likelihood."""
        z = self.logits(x_val)
        best = (np.inf, 1.0)
        for T in np.exp(np.linspace(np.log(0.3), np.log(10), 80)):
            p = self._softmax(z / T)
            nll = -np.log(p[np.arange(len(y_val)), y_val] + 1e-12).mean()
            if nll < best[0]:
                best = (nll, float(T))
        self.temperature = best[1]
        return self.temperature

    def save(self, path):
        np.savez_compressed(path, temperature=self.temperature, mean=self.mean, std=self.std, **self.p)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls.__new__(cls)
        net.p = {k: d[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")}
        net.temperature, net.mean, net.std = float(d["temperature"]), d["mean"], d["std"]
        return net


def augment_batch(xb, rng):
    """Small shifts and brightness/contrast changes."""
    out = np.empty_like(xb)
    for i, im in enumerate(xb):
        dy, dx = rng.integers(-2, 3, 2)
        im = np.roll(np.roll(im, dy, 0), dx, 1)
        im = np.clip((im - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1), 0, 1)
        out[i] = im
    return out
