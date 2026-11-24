"""GTSRB loading (official, torchvision or Kaggle layouts) and synthetic signs for tests."""

from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path

import numpy as np

from .signs import N_CLASSES, SIGNS

EXTS = (".ppm", ".png", ".jpg", ".jpeg")


def _load_image(path, size, roi=None):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if roi is not None:
        im = im.crop(roi)
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def _read_csv(path):
    with open(path, newline="") as f:
        head = f.readline()
        f.seek(0)
        delim = ";" if head.count(";") > head.count(",") else ","
        return list(csv.DictReader(f, delimiter=delim, skipinitialspace=True))


def _roi(row):
    keys = [("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")]
    for k in keys:
        if all(x in row for x in k):
            try:
                return tuple(int(row[x]) for x in k)
            except ValueError:
                return None
    return None


def load_gtsrb(root: str | Path, size: int = 32, max_per_class: int | None = None, seed: int = 0):
    """Return (x_train, y_train, x_test, y_test). Training images are found in folders named
    by class id (0..42 or 00000..00042); test images come with a CSV holding a ClassId column
    (GT-final_test.csv or Test.csv)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    class_dirs = {}
    for d in glob.glob(str(root / "**"), recursive=True):
        name = os.path.basename(d.rstrip("/"))
        if os.path.isdir(d) and re.fullmatch(r"0*\d{1,2}", name) and "test" not in d.lower():
            cid = int(name)
            if cid < N_CLASSES and any(f.lower().endswith(EXTS) for f in os.listdir(d)):
                class_dirs.setdefault(cid, d)
    if len(class_dirs) < N_CLASSES:
        raise SystemExit(f"Found training folders for {len(class_dirs)} of 43 classes under {root}. "
                         "Unzip the GTSRB training set there (see the README).")
    for cid, d in sorted(class_dirs.items()):
        rois = {}
        for c in glob.glob(os.path.join(d, "*.csv")):
            for row in _read_csv(c):
                fn = row.get("Filename") or row.get("Path")
                if fn:
                    rois[os.path.basename(fn)] = _roi(row)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))
        if max_per_class and len(files) > max_per_class:
            files = sorted(rng.choice(files, max_per_class, replace=False))
        for f in files:
            xs.append(_load_image(os.path.join(d, f), size, rois.get(f)))
            ys.append(cid)
    x_tr, y_tr = np.stack(xs), np.array(ys)

    test_csv = None
    for pat in ("**/GT-final_test.csv", "**/Test.csv"):
        hits = glob.glob(str(root / pat), recursive=True)
        if hits:
            test_csv = hits[0]
            break
    if test_csv is None:
        print("no test labels found: holding out 20% of training images as the test set")
        idx = rng.permutation(len(x_tr))
        cut = int(0.8 * len(idx))
        return x_tr[idx[:cut]], y_tr[idx[:cut]], x_tr[idx[cut:]], y_tr[idx[cut:]]
    rows = _read_csv(test_csv)
    base = Path(test_csv).parent
    img_dirs = [base] + [Path(p) for p in glob.glob(str(root / "**/Final_Test/Images"), recursive=True)]
    xt, yt = [], []
    for row in rows:
        fn = row.get("Filename") or row.get("Path")
        path = None
        for d in img_dirs + [root]:
            for cand in (d / fn, d / os.path.basename(fn)):
                if cand.exists():
                    path = cand
                    break
            if path:
                break
        if path is None:
            continue
        xt.append(_load_image(path, size, _roi(row)))
        yt.append(int(row["ClassId"]))
    if not xt:
        raise SystemExit(f"Test CSV {test_csv} found but none of its images; check the folder layout.")
    return x_tr, y_tr, np.stack(xt), np.array(yt)


# --- synthetic signs (tests and smoke runs only) ----------------------------------------

def _shape_mask(kind, size):
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1) * 2 - 1
    r = np.sqrt(xx ** 2 + yy ** 2)
    if kind in ("triangle",):
        return (yy > -0.8) & (yy < 0.8) & (np.abs(xx) < (yy + 0.8) * 0.6)
    if kind == "inv_triangle":
        return (yy > -0.8) & (yy < 0.8) & (np.abs(xx) < (0.8 - yy) * 0.6)
    if kind == "octagon":
        return (np.abs(xx) < 0.8) & (np.abs(yy) < 0.8) & (np.abs(xx) + np.abs(yy) < 1.15)
    if kind == "diamond":
        return np.abs(xx) + np.abs(yy) < 0.85
    return r < 0.8


def synthetic_signs(n_per_class: int = 60, size: int = 32, seed: int = 0):
    """Coloured shapes with a class-specific inner glyph, noise, shifts and lighting changes."""
    rng = np.random.default_rng(seed)
    style = {"speed": ("circle", (0.85, 0.1, 0.1), (1, 1, 1)), "end80": ("circle", (0.4, 0.4, 0.4), (1, 1, 1)),
             "end_all": ("circle", (0.4, 0.4, 0.4), (1, 1, 1)), "end_nopass": ("circle", (0.4, 0.4, 0.4), (1, 1, 1)),
             "nopass": ("circle", (0.85, 0.1, 0.1), (1, 1, 1)), "no_entry": ("circle", (0.85, 0.1, 0.1), (0.85, 0.1, 0.1)),
             "stop": ("octagon", (0.85, 0.1, 0.1), (0.85, 0.1, 0.1)), "yield": ("inv_triangle", (0.85, 0.1, 0.1), (1, 1, 1)),
             "junction": ("triangle", (0.85, 0.1, 0.1), (1, 1, 1)), "warning": ("triangle", (0.85, 0.1, 0.1), (1, 1, 1)),
             "people": ("triangle", (0.85, 0.1, 0.1), (1, 1, 1)), "neutral": ("circle", (0.1, 0.3, 0.8), (0.1, 0.3, 0.8))}
    xs, ys = [], []
    for s in SIGNS:
        shape, ring, fill = style[s.kind]
        if s.cid == 12:
            shape, ring, fill = "diamond", (1, 1, 1), (0.95, 0.8, 0.1)
        glyph_rng = np.random.default_rng(1000 + s.cid)
        glyph = glyph_rng.random((4, 4)) > 0.5
        outer = _shape_mask(shape, size)
        inner = _shape_mask(shape, size + 12)[6:-6, 6:-6]
        for _ in range(n_per_class):
            bg = rng.uniform(0.2, 0.7, 3)
            im = np.ones((size, size, 3), np.float32) * bg
            im[outer] = ring
            im[inner] = fill
            g = np.kron(glyph, np.ones((3, 3)))
            y0 = (size - g.shape[0]) // 2
            region = im[y0:y0 + g.shape[0], y0:y0 + g.shape[1]]
            region[g.astype(bool)] = 0.05
            im = np.roll(np.roll(im, rng.integers(-2, 3), 0), rng.integers(-2, 3), 1)
            im = np.clip(im * rng.uniform(0.6, 1.2) + rng.normal(0, 0.04, im.shape), 0, 1)
            xs.append(im.astype(np.float32))
            ys.append(s.cid)
    x, y = np.stack(xs), np.array(ys)
    idx = rng.permutation(len(x))
    return x[idx], y[idx]
