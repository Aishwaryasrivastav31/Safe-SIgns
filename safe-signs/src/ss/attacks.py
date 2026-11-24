"""Attacks on the perception network (all white-box, pixel values in [0, 1]).

  fgsm         one signed-gradient step of size eps
  pgd          projected gradient descent in an L-infinity ball of radius eps
  patch        an unbounded sticker: PGD restricted to a square patch
  pgd_target   targeted PGD towards a chosen class (used for the dangerous and the
               rule-aware attacks)
"""

from __future__ import annotations

import numpy as np


def fgsm(net, x, y, eps):
    return np.clip(x + eps * np.sign(net.loss_and_input_grad(x, y)), 0, 1).astype(np.float32)


def pgd(net, x, y, eps, steps=10, step=None, target=None, mask=None, seed=0, batch=256):
    rng = np.random.default_rng(seed)
    step = step or eps / 4
    out = np.empty_like(x)
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        tb = None if target is None else target[i:i + batch]
        mb = 1.0 if mask is None else mask[i:i + batch]
        adv = np.clip(xb + mb * rng.uniform(-eps, eps, xb.shape).astype(np.float32), 0, 1)
        for _ in range(steps):
            if tb is None:
                g = net.loss_and_input_grad(adv, yb)
            else:
                g = net.loss_and_input_grad(adv, tb, targeted=True)
            adv = adv + mb * step * np.sign(g)
            adv = np.clip(np.clip(adv, xb - eps, xb + eps), 0, 1).astype(np.float32)
        out[i:i + batch] = adv
    return out


def patch_masks(n, size=32, patch=8, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((n, size, size, 1), np.float32)
    for i in range(n):
        y0, x0 = rng.integers(4, size - patch - 4, 2)
        m[i, y0:y0 + patch, x0:x0 + patch] = 1
    return m
