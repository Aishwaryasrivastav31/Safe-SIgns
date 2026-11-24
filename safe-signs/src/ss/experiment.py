"""The full study: train the network, attack it, and compare four decision systems."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .attacks import fgsm, patch_masks, pgd
from .cnn import CNN
from .contexts import corrupt, sample_context
from .logic import MAX_LEGAL, consistency_mask, consistent, decide, is_unsafe, required_speed
from .signs import N_CLASSES
from .verify import SYSTEMS, check, check_envelope


@dataclass
class Config:
    epochs: int = 8
    batch: int = 128
    lr: float = 1e-3
    n_eval: int = 3000
    eps: float = 8 / 255
    pgd_steps: int = 10
    patch_size: int = 8
    patch_steps: int = 20
    tau: float = 0.6
    plausible_eps: float = 0.02
    context_noise: list = field(default_factory=lambda: [0.0, 0.1, 0.2])
    n_boot: int = 1000
    seed: int = 0
    adv_training: bool = True          # also train a PGD adversarially trained network as a baseline
    adv_epochs: int = 6
    adv_steps: int = 3
    tau_grid: list = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    soft_alpha: float = 20.0     # soft rules: smoothing strength
    soft_prior: float = 0.01     # soft rules: violation rate assumed for a rule with no real data
    extra_seeds: list = field(default_factory=lambda: [1, 2])   # more training seeds for the standard CNN
    # round four: stronger attacks and analysis
    n_stress: int = 600              # images used for the adaptive, map-spoof and eps-sweep attacks
    adaptive_steps: int = 30         # PGD steps per target for the rule-aware adaptive attack
    adaptive_targets: int = 3        # how many rule-consistent unsafe targets it tries per image
    eps_sweep: list = field(default_factory=lambda: [2, 4, 8, 16])   # in pixel levels out of 255
    envelope_random: int = 1000      # random probability vectors per context in the envelope check


def most_dangerous_target(y, ctxs, consistent_only: bool):
    """For each image, the sign that would let the car go fastest in its context
    (optionally only among signs the rules accept)."""
    t = np.empty(len(y), int)
    for i, (yi, c) in enumerate(zip(y, ctxs)):
        mask = consistency_mask(c) if consistent_only else np.ones(N_CLASSES, bool)
        mask[yi] = False
        cands = np.where(mask)[0]
        speeds = np.array([required_speed(k, c) for k in cands])
        t[i] = cands[np.argmax(speeds)]
    return t


def evaluate(probs, y, ctx_seen, ctx_true, cfg) -> dict:
    """Run all four systems; return per-image arrays."""
    out = {}
    for name, (logic, fb) in SYSTEMS.items():
        dec = [decide(p, c, logic, fb, cfg.tau, cfg.plausible_eps) for p, c in zip(probs, ctx_seen)]
        cid = np.array([d.cid if d.cid is not None else -1 for d in dec])
        speed = np.array([d.speed for d in dec])
        unsafe = np.array([is_unsafe(s, yi, ct) for s, yi, ct in zip(speed, y, ctx_true)])
        need = np.array([required_speed(yi, ct) for yi, ct in zip(y, ctx_true)])
        out[name] = {"correct": cid == y, "unsafe": unsafe, "fallback": np.array([d.fallback for d in dec]),
                     "overspeed": np.where(unsafe, speed - need, 0.0),
                     "rejected_mass": np.array([d.rejected_mass for d in dec]),
                     "cid": cid, "speed": speed,
                     "acted_impossible": np.array([c >= 0 and not consistent(int(c), cs) for c, cs in zip(cid, ctx_seen)]),
                     "over_legal": np.array([s > MAX_LEGAL[cs.road] + 1e-9 for s, cs in zip(speed, ctx_seen)])}
    return out


def boot_diff(a, b, n_boot, seed):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        d.append(a[i].mean() - b[i].mean())
    return float(a.mean() - b.mean()), *map(float, np.quantile(d, [0.025, 0.975]))


def mcnemar(a, b):
    """Exact McNemar on paired binary outcomes (a only vs b only)."""
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def make_attacks(net, x, y, ctxs, cfg):
    return {"clean": x,
            "FGSM": fgsm(net, x, y, cfg.eps),
            "PGD": pgd(net, x, y, cfg.eps, cfg.pgd_steps, seed=cfg.seed),
            "patch": pgd(net, x, y, 1.0, cfg.patch_steps, step=0.1,
                         mask=patch_masks(len(x), x.shape[1], cfg.patch_size, cfg.seed), seed=cfg.seed),
            "dangerous (targeted)": pgd(net, x, y, cfg.eps, cfg.pgd_steps, seed=cfg.seed,
                                        target=most_dangerous_target(y, ctxs, False)),
            "rule-aware (targeted)": pgd(net, x, y, cfg.eps, cfg.pgd_steps, seed=cfg.seed,
                                         target=most_dangerous_target(y, ctxs, True))}


def summarise(per_image, model_name):
    rows = []
    for att, res in per_image.items():
        for sysname, r in res.items():
            rows.append({"model": model_name, "attack": att, "system": sysname, "accuracy": float(r["correct"].mean()),
                         "unsafe_rate": float(r["unsafe"].mean()), "fallback_rate": float(r["fallback"].mean()),
                         "mean_overspeed_when_unsafe": float(r["overspeed"][r["unsafe"]].mean()) if r["unsafe"].any() else 0.0})
    return rows


def run(x_tr, y_tr, x_te, y_te, cfg: Config, out: Path, net: CNN | None = None,
        adv_net: CNN | None = None, pool: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(x_tr))
    n_val = max(200, len(idx) // 10)
    val, tr = idx[:n_val], idx[n_val:]
    if net is None:
        print(f"training the CNN on {len(tr)} images ({cfg.epochs} epochs)")
        net = CNN(seed=cfg.seed).fit(x_tr[tr], y_tr[tr], x_tr[val], y_tr[val], epochs=cfg.epochs,
                                     batch=cfg.batch, lr=cfg.lr, seed=cfg.seed)
        T = net.calibrate_temperature(x_tr[val], y_tr[val])
        print(f"  temperature after calibration: {T:.2f}")
        net.save(out / "cnn_weights.npz")
    if cfg.adv_training and adv_net is None:
        print(f"adversarial training (PGD-{cfg.adv_steps}, eps {cfg.eps * 255:.0f}/255) for {cfg.adv_epochs} epochs")
        adv_net = CNN(seed=cfg.seed + 1).fit(x_tr[tr], y_tr[tr], x_tr[val], y_tr[val], epochs=cfg.adv_epochs,
                                             batch=cfg.batch, lr=cfg.lr, seed=cfg.seed, adv_eps=cfg.eps,
                                             adv_steps=cfg.adv_steps)
        adv_net.calibrate_temperature(x_tr[val], y_tr[val])
        adv_net.save(out / "cnn_adv_weights.npz")

    ev = rng.choice(len(x_te), min(cfg.n_eval, len(x_te)), replace=False)
    x, y = x_te[ev], y_te[ev]
    ctxs = [sample_context(int(c), rng, pool=pool) for c in y]
    n_real = sum(1 for c in y if pool and int(c) in pool)
    true_inconsistent = float(np.mean([not consistency_mask(c)[yi] for c, yi in zip(ctxs, y)]))
    test_acc = float((net.predict_proba(x_te).argmax(1) == y_te).mean())
    print(f"clean test accuracy (all {len(y_te)} test images): {test_acc:.3f}")
    if pool:
        print(f"contexts from OpenStreetMap for {n_real}/{len(y)} images; true sign breaks a rule in "
              f"{true_inconsistent:.1%} of contexts")

    print("attacking the standard network")
    adv = make_attacks(net, x, y, ctxs, cfg)
    probs = {k: net.predict_proba(v) for k, v in adv.items()}
    per_image = {att: evaluate(p, y, ctxs, ctxs, cfg) for att, p in probs.items()}
    rows = summarise(per_image, "standard")

    adv_info = {}
    if adv_net is not None:
        adv_acc = float((adv_net.predict_proba(x_te).argmax(1) == y_te).mean())
        print(f"adversarially trained network: clean test accuracy {adv_acc:.3f}; attacking it")
        adv_b = make_attacks(adv_net, x, y, ctxs, cfg)
        probs_b = {k: adv_net.predict_proba(v) for k, v in adv_b.items()}
        per_image_b = {att: evaluate(p, y, ctxs, ctxs, cfg) for att, p in probs_b.items()}
        rows += summarise(per_image_b, "adversarially trained")
        adv_info = {"clean_test_accuracy": adv_acc,
                    "attack_success": {k: float((p.argmax(1) != y).mean()) for k, p in probs_b.items()}}

    # Paired comparisons: full system vs the two ablations.
    comps = []
    for att in probs:
        full = per_image[att]["logic + fallback"]["unsafe"]
        others = [("standard", o, per_image[att][o]["unsafe"])
                  for o in ("network alone", "network + fallback", "logic only", "soft logic + fallback")]
        if adv_net is not None:
            others.append(("adversarially trained", "network alone", per_image_b[att]["network alone"]["unsafe"]))
            others.append(("adversarially trained", "logic + fallback", per_image_b[att]["logic + fallback"]["unsafe"]))
        for model, other, o in others:
            d, lo, hi = boot_diff(full.astype(float), o.astype(float), cfg.n_boot, cfg.seed)
            n01, n10, p = mcnemar(full, o)
            comps.append({"attack": att, "vs": f"{other} ({model})", "unsafe_diff": d, "ci_low": lo, "ci_high": hi,
                          "full_only_unsafe": n01, "other_only_unsafe": n10, "mcnemar_p": p})

    # How strict should the fallback be? Safety vs how often the car slows down.
    tau_rows = []
    for att in ("clean", "PGD", "dangerous (targeted)"):
        for tau in cfg.tau_grid:
            for sysname, (logic, fb) in (("network + fallback", (False, True)), ("logic + fallback", (True, True)),
                                         ("soft logic + fallback", ("soft", True))):
                dec = [decide(p, c, logic, fb, tau, cfg.plausible_eps) for p, c in zip(probs[att], ctxs)]
                speed = np.array([d.speed for d in dec])
                unsafe = np.array([is_unsafe(sp, yi, ct) for sp, yi, ct in zip(speed, y, ctxs)])
                tau_rows.append({"attack": att, "tau": tau, "system": sysname, "unsafe_rate": float(unsafe.mean()),
                                 "fallback_rate": float(np.mean([d.fallback for d in dec]))})

    # Training-seed variation: retrain the standard CNN with other seeds, same images and contexts.
    seed_rows = [dict(r, seed=cfg.seed) for r in rows if r["model"] == "standard"]
    for sd in cfg.extra_seeds:
        wpath = out / f"cnn_weights_s{sd}.npz"
        if wpath.exists():
            net_s = CNN.load(wpath)
        else:
            print(f"training the CNN with seed {sd}")
            net_s = CNN(seed=sd).fit(x_tr[tr], y_tr[tr], x_tr[val], y_tr[val], epochs=cfg.epochs, batch=cfg.batch,
                                     lr=cfg.lr, seed=sd)
            net_s.calibrate_temperature(x_tr[val], y_tr[val])
            net_s.save(wpath)
        print(f"attacking the seed-{sd} network")
        adv_s = make_attacks(net_s, x, y, ctxs, cfg)
        per_s = {att: evaluate(net_s.predict_proba(v), y, ctxs, ctxs, cfg) for att, v in adv_s.items()}
        seed_rows += [dict(r, seed=sd) for r in summarise(per_s, "standard")]
    seeds_df = None
    if cfg.extra_seeds:
        import pandas as pd
        sr = pd.DataFrame(seed_rows)
        seeds_df = (sr.groupby(["attack", "system"], sort=False)
                      .agg(n_seeds=("seed", "nunique"), unsafe_mean=("unsafe_rate", "mean"), unsafe_std=("unsafe_rate", "std"),
                           fallback_mean=("fallback_rate", "mean"), accuracy_mean=("accuracy", "mean")).reset_index())

    # Wrong maps: corrupt the context the system sees (truth unchanged).
    noise_rows = []
    for rho in cfg.context_noise:
        nr = np.random.default_rng(cfg.seed + 7)
        seen = [corrupt(c, nr) if nr.random() < rho else c for c in ctxs]
        for att in ("clean", "PGD"):
            res = evaluate(probs[att], y, seen, ctxs, cfg)
            for sysname, r in res.items():
                noise_rows.append({"context_noise": rho, "attack": att, "system": sysname,
                                   "accuracy": float(r["correct"].mean()), "unsafe_rate": float(r["unsafe"].mean()),
                                   "fallback_rate": float(r["fallback"].mean())})

    # Rule violations as an attack detector: mass the network puts on impossible signs.
    rm_clean = per_image["clean"]["logic + fallback"]["rejected_mass"]
    detect = []
    for att in probs:
        if att == "clean":
            continue
        rm = per_image[att]["logic + fallback"]["rejected_mass"]
        lab = np.r_[np.zeros(len(rm_clean)), np.ones(len(rm))]
        detect.append({"attack": att, "auroc": float(roc_auc_score(lab, np.r_[rm_clean, rm])),
                       "median_rejected_mass_clean": float(np.median(rm_clean)),
                       "median_rejected_mass_attacked": float(np.median(rm))})

    print("exhaustive check of the decision layer")
    ver = check(cfg.tau, cfg.plausible_eps)
    print("envelope check (random network outputs)")
    env_ver = check_envelope(cfg.tau, cfg.plausible_eps, cfg.envelope_random, cfg.seed)

    # ---- round four: analysis of the main run
    from . import analysis as AN
    by_model = {"standard": per_image}
    if adv_net is not None:
        by_model["adversarially trained"] = per_image_b
    cert_summary, cert_check = AN.certification(y, ctxs, ctxs, per_image)
    causes, helped, breakdown = AN.failure_analysis(per_image, probs, y, ctxs, ctxs)
    import pandas as pd
    main_df = pd.DataFrame(rows)
    analysis = {"baselines": AN.baselines_table(main_df), "baselines_accuracy": AN.baselines_table(main_df, "accuracy"),
                "tradeoff": AN.tradeoff_table(main_df), "runtime_violations": AN.runtime_violations(by_model),
                "certification_check": cert_check, "failure_causes": causes, "rules_that_helped": helped,
                "failure_breakdown": breakdown}

    # ---- round four: stronger attacks on a subset (adaptive to the rules and fallback, map spoofing, eps sweep)
    stress = run_stress(net, adv_net, x, y, ctxs, cfg)
    analysis.update({k: v for k, v in stress.items() if isinstance(v, pd.DataFrame)})

    import pandas as pd
    res = {"main": pd.DataFrame(rows), "comparisons": pd.DataFrame(comps), "noise": pd.DataFrame(noise_rows),
           "detect": pd.DataFrame(detect), "tau": pd.DataFrame(tau_rows), "adv": adv_info, "seeds": seeds_df,
           "context_source": "OpenStreetMap" if pool else "simulated", "real_context_images": n_real,
           "true_sign_breaks_rule": true_inconsistent, "verify": ver, "test_accuracy": test_acc,
           "temperature": net.temperature, "n_eval": int(len(y)),
           "attack_success": {k: float((p.argmax(1) != y).mean()) for k, p in probs.items()},
           "examples": (x, y, ctxs, adv, probs)}
    res.update({"analysis": analysis, "certification": cert_summary, "envelope_verify": env_ver})
    for k in ("main", "comparisons", "noise", "detect", "tau", "seeds"):
        if res[k] is not None:
            res[k].to_csv(out / f"{k}.csv", index=False)
    for k, v in analysis.items():
        v.to_csv(out / f"{k}.csv", index=False)
    (out / "certification.json").write_text(json.dumps(cert_summary, indent=2))
    (out / "envelope_verification.json").write_text(json.dumps(env_ver, indent=2, default=str))
    (out / "verification.json").write_text(json.dumps(ver, indent=2, default=str))
    (out / "info.json").write_text(json.dumps({"test_accuracy": test_acc, "temperature": net.temperature,
                                               "n_eval": int(len(y)), "attack_success": res["attack_success"],
                                               "adversarially_trained": adv_info, "context_source": res["context_source"],
                                               "true_sign_breaks_rule": true_inconsistent}, indent=2))
    return res


def run_stress(net, adv_net, x, y, ctxs, cfg) -> dict:
    """Attacks designed to beat the full system, on the first n_stress evaluation images."""
    from . import analysis as AN
    from .adaptive import eps_sweep, rule_aware_adaptive, worst_spoof

    import pandas as pd

    n = min(cfg.n_stress, len(x))
    xs, ys, cs = x[:n], y[:n], ctxs[:n]
    spoofed = [worst_spoof(int(yi), c) for yi, c in zip(ys, cs)]
    models = {"standard": net} if adv_net is None else {"standard": net, "adversarially trained": adv_net}
    rows, rank_rows, failure = [], [], []
    for mname, m in models.items():
        print(f"stress attacks on the {mname} network ({n} images)")
        adv_a, rank_a = rule_aware_adaptive(m, xs, ys, cs, cs, cfg.eps, cfg.adaptive_steps, cfg.adaptive_targets,
                                            cfg.tau, cfg.plausible_eps, cfg.seed)
        adv_s, rank_s = rule_aware_adaptive(m, xs, ys, cs, spoofed, cfg.eps, cfg.adaptive_steps,
                                            cfg.adaptive_targets, cfg.tau, cfg.plausible_eps, cfg.seed)
        scen = {"rule-aware adaptive": (adv_a, cs), "map spoof only (clean image)": (xs, spoofed),
                "map spoof + rule-aware adaptive": (adv_s, spoofed)}
        per = {k: evaluate(m.predict_proba(v[0]), ys, v[1], cs, cfg) for k, v in scen.items()}
        rows += summarise(per, mname)
        for k, rk in (("rule-aware adaptive", rank_a), ("map spoof + rule-aware adaptive", rank_s)):
            rank_rows.append({"model": mname, "attack": k, "certified_no_target": float((rk == -2).mean()),
                              "attack_failed": float((rk == -1).mean()),
                              **{f"won_with_target_{r + 1}": float((rk == r).mean()) for r in range(cfg.adaptive_targets)}})
        probs_s = {k: m.predict_proba(v[0]) for k, v in scen.items()}
        cz, _, _ = AN.failure_analysis(per, probs_s, ys, cs, {k: v[1] for k, v in scen.items()})
        failure.append(cz.assign(model=mname))
    sweep = eps_sweep(net, xs, ys, cs, [e / 255 for e in cfg.eps_sweep], cfg.pgd_steps, cfg.tau, cfg.plausible_eps,
                      cfg.seed, {k: v for k, v in SYSTEMS.items() if k != "logic only"})
    return {"stress": pd.DataFrame(rows), "adaptive_targets": pd.DataFrame(rank_rows),
            "stress_failure_causes": pd.concat(failure, ignore_index=True), "eps_sweep": pd.DataFrame(sweep)}
