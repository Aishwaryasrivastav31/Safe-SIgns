"""Run the study.

    python scripts/run.py --smoke    synthetic signs, a few minutes, no download
    python scripts/run.py            GTSRB, settings in configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from ss.cnn import CNN  # noqa: E402
from ss.data import load_gtsrb, synthetic_signs  # noqa: E402
from ss.experiment import Config, run  # noqa: E402
from ss.logic import to_prolog  # noqa: E402
from ss.osm import load_pool  # noqa: E402
from ss.logic import set_rule_weights, weights_from_reliability  # noqa: E402
from ss.report import figures, summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse-weights", action="store_true", help="skip training if cnn_weights.npz exists")
    ap.add_argument("--no-seeds", action="store_true", help="skip the extra training seeds (saves about an hour)")
    args = ap.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    t0 = time.time()
    if args.smoke:
        x, y = synthetic_signs(70, seed=0)
        cut = int(0.8 * len(x))
        x_tr, y_tr, x_te, y_te = x[:cut], y[:cut], x[cut:], y[cut:]
        cfg = Config(epochs=6, n_eval=500, n_boot=200, adv_epochs=3, extra_seeds=[1], n_stress=200, adaptive_steps=15,
                     envelope_random=100, eps_sweep=[4, 8, 16])
        out = ROOT / "results" / "smoke"
        note = "**Synthetic signs (smoke test).** These numbers only show that the code runs; they say nothing about real traffic signs."
    else:
        x_tr, y_tr, x_te, y_te = load_gtsrb(ROOT / c["data_dir"], c["image_size"], c.get("max_per_class"), c["seed"])
        print(f"GTSRB: {len(x_tr)} training and {len(x_te)} test images")
        cfg = Config(**{k: c[k] for k in ("epochs", "batch", "lr", "n_eval", "pgd_steps", "patch_size",
                                          "patch_steps", "tau", "plausible_eps", "context_noise", "n_boot", "seed",
                                          "adv_training", "adv_epochs", "adv_steps", "tau_grid", "extra_seeds",
                                          "soft_alpha", "soft_prior", "n_stress", "adaptive_steps",
                                          "adaptive_targets", "eps_sweep", "envelope_random") if k in c},
                     eps=c["eps_255"] / 255)
        out = ROOT / c["output_dir"]
        note = "Data: German Traffic Sign Recognition Benchmark (GTSRB). "
    if args.no_seeds:
        cfg.extra_seeds = []
    out.mkdir(parents=True, exist_ok=True)
    (out / "rules.pl").write_text(to_prolog())
    net = CNN.load(out / "cnn_weights.npz") if args.reuse_weights and (out / "cnn_weights.npz").exists() else None
    adv_net = CNN.load(out / "cnn_adv_weights.npz") if args.reuse_weights and (out / "cnn_adv_weights.npz").exists() else None
    if args.reuse_weights:
        for f in ("cnn_weights.npz", "cnn_adv_weights.npz"):
            print(("reusing " if (out / f).exists() else "NOT FOUND, will train: ") + str(out / f))
    pool, osm_rel = None, None
    pool_path = ROOT / "data" / "osm_contexts.json"
    if not args.smoke and pool_path.exists():
        pool = load_pool(pool_path)
        print(f"using OpenStreetMap contexts for {len(pool)} classes")
        rel_path = ROOT / "data" / "osm_rule_reliability.csv"
        if rel_path.exists():
            import pandas as pd
            osm_rel = pd.read_csv(rel_path)
    rows = osm_rel.to_dict("records") if osm_rel is not None else []
    if not rows:  # no real data: every rule gets the prior, i.e. nearly hard
        from ss.logic import RULES
        rows = [{"rule": n, "real_signs_it_applies_to": 0, "violated_on_real_map": 0} for n, _, _ in RULES]
    weights = weights_from_reliability(rows, cfg.soft_alpha, cfg.soft_prior)
    set_rule_weights(weights)
    res = run(x_tr, y_tr, x_te, y_te, cfg, out, net, adv_net, pool)
    res["osm_reliability"] = osm_rel
    res["rule_weights"] = weights
    figs = figures(res, out / "figures")
    p = summary(res, out, figs, note)
    print(p.read_text())
    print(f"done in {time.time() - t0:.0f} s; results in {out}")


if __name__ == "__main__":
    main()
