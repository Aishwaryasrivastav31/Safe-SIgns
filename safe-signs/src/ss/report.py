"""Figures and summary. Every number comes from the experiment output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .logic import decide, explain, required_speed
from .signs import NAMES
from .verify import SYSTEMS

COL = {"network alone": "#9e9e9e", "network + fallback": "#5b8def", "logic only": "#f0a030", "logic + fallback": "#d1495b",
       "soft logic + fallback": "#2a9d8f"}


def _save(fig, p):
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)


def figures(res, d: Path) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    names = []
    main_all = res["main"]
    main = main_all[main_all["model"] == "standard"]
    attacks = list(dict.fromkeys(main["attack"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, ttl in zip(axes, ("unsafe_rate", "fallback_rate"),
                               ("Unsafe decisions (faster than the real sign allows)", "Cautious fallbacks")):
        w = 0.8 / len(SYSTEMS)
        for i, s in enumerate(SYSTEMS):
            vals = [main[(main.attack == a) & (main.system == s)][metric].iloc[0] for a in attacks]
            ax.bar(np.arange(len(attacks)) + (i - (len(SYSTEMS) - 1) / 2) * w, vals, width=w, color=COL[s], label=s)
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels(attacks, rotation=20, ha="right", fontsize=8)
        ax.set_title(ttl)
    axes[0].set_ylabel("Share of test images")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, d / "unsafe_by_attack.png"); names.append("unsafe_by_attack.png")

    # Standard vs adversarially trained network
    if (main_all["model"] == "adversarially trained").any():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        combos = [("standard", "network alone", "#9e9e9e"), ("adversarially trained", "network alone", "#5b8def"),
                  ("standard", "logic + fallback", "#f0a030"), ("adversarially trained", "logic + fallback", "#d1495b")]
        w = 0.2
        for i, (m, sname, col) in enumerate(combos):
            vals = [main_all[(main_all.model == m) & (main_all.attack == a) & (main_all.system == sname)]["unsafe_rate"].iloc[0]
                    for a in attacks]
            ax.bar(np.arange(len(attacks)) + (i - 1.5) * w, vals, width=w, color=col, label=f"{m}: {sname}")
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels(attacks, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Unsafe decisions")
        ax.set_title("Adversarial training vs logic, and both together")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, d / "adversarial_training.png"); names.append("adversarial_training.png")

    # Fallback threshold: safety against how often the car slows down
    tt = res.get("tau")
    if tt is not None and len(tt):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, att in zip(axes, ("clean", "PGD", "dangerous (targeted)")):
            for sname in ("network + fallback", "logic + fallback", "soft logic + fallback"):
                g = tt[(tt.attack == att) & (tt.system == sname)].sort_values("tau")
                ax.plot(g["fallback_rate"], g["unsafe_rate"], marker="o", color=COL[sname], label=sname)
                for _, r in g.iterrows():
                    ax.annotate(f"{r['tau']:.2f}", (r["fallback_rate"], r["unsafe_rate"]), fontsize=6,
                                xytext=(3, 3), textcoords="offset points")
            ax.set_title(att)
            ax.set_xlabel("Share of images with a cautious fallback")
        axes[0].set_ylabel("Unsafe decisions")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Choosing the fallback threshold (labels are tau)")
        _save(fig, d / "fallback_threshold.png"); names.append("fallback_threshold.png")

    nz = res["noise"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, att in zip(axes, ("clean", "PGD")):
        for s in SYSTEMS:
            g = nz[(nz.attack == att) & (nz.system == s)]
            ax.plot(g["context_noise"], g["unsafe_rate"], marker="o", color=COL[s], label=s)
        ax.set_title(f"{att} images")
        ax.set_xlabel("Share of wrong map facts")
    axes[0].set_ylabel("Unsafe decisions")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("When the logic's knowledge is wrong")
    _save(fig, d / "wrong_map.png"); names.append("wrong_map.png")

    ex_x, ex_y, ctxs, adv, probs = res["examples"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for att, col in (("clean", "#5b8def"), ("PGD", "#d1495b"), ("rule-aware (targeted)", "#333333")):
        rm = np.array([decide(p, c, True, True).rejected_mass for p, c in zip(probs[att], ctxs)])
        ax.hist(rm, bins=np.linspace(0, 1, 26), alpha=0.5, color=col, label=att)
    ax.set_yscale("log")
    ax.set_xlabel("Probability the network gives to signs the rules call impossible")
    ax.set_ylabel("Images (log scale)")
    ax.set_title("Rule violations as an alarm")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, d / "rule_violation_alarm.png"); names.append("rule_violation_alarm.png")

    an = res.get("analysis") or {}
    sw = an.get("eps_sweep")
    if sw is not None and len(sw):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, att in zip(axes, ("PGD", "rule-aware (targeted)")):
            for sname in dict.fromkeys(sw.system):
                g = sw[(sw.attack == att) & (sw.system == sname)].sort_values("eps_255")
                ax.plot(g["eps_255"], g["unsafe_rate"], marker="o", color=COL.get(sname, "#333"), label=sname)
            ax.set_title(att)
            ax.set_xlabel("Attack budget eps (pixel levels out of 255)")
        axes[0].set_ylabel("Unsafe decisions")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Attack strength curve")
        _save(fig, d / "eps_sweep.png"); names.append("eps_sweep.png")

    fc = an.get("failure_causes")
    st = an.get("stress_failure_causes")
    if fc is not None and len(fc):
        parts = [fc.assign(model="standard")]
        if st is not None and len(st):
            parts.append(st[st.model == "standard"])
        allc = pd.concat(parts, ignore_index=True)
        atts = list(dict.fromkeys(allc.attack))
        causes = ["inside the rule envelope", "fallback still too fast", "true sign breaks a rule", "map wrong"]
        ccol = {"inside the rule envelope": "#d1495b", "fallback still too fast": "#f0a030",
                "true sign breaks a rule": "#5b8def", "map wrong": "#333333"}
        fig, ax = plt.subplots(figsize=(11, 4.5))
        bottom = np.zeros(len(atts))
        for c in causes:
            v = np.array([allc[(allc.attack == a) & (allc.cause == c)].share_of_all_images.sum() for a in atts])
            ax.bar(range(len(atts)), v, bottom=bottom, color=ccol[c], label=c)
            bottom += v
        ax.set_xticks(range(len(atts)))
        ax.set_xticklabels(atts, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Share of images with an unsafe decision")
        ax.set_title("Why the full system is still unsafe (standard network)")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, d / "failure_causes.png"); names.append("failure_causes.png")

    to = an.get("tradeoff")
    if to is not None and len(to):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for _, r in to.iterrows():
            mk = "o" if r["model"] == "standard" else "s"
            ax.scatter(r["clean_unsafe"], r["worst_attack_unsafe"], color=COL.get(r["system"], "#333"), marker=mk, s=60)
            ax.annotate(f"{r['system']} ({'std' if r['model'] == 'standard' else 'adv'})",
                        (r["clean_unsafe"], r["worst_attack_unsafe"]), fontsize=6, xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel("Unsafe decisions on clean images")
        ax.set_ylabel("Unsafe decisions under the worst attack")
        ax.set_title("Clean cost vs worst-case robustness (circle: standard, square: adv. trained)")
        _save(fig, d / "tradeoff.png"); names.append("tradeoff.png")

    # Worked examples: images where the network alone is unsafe under PGD.
    pa, pf = probs["PGD"], None
    picks = []
    for i, (p, c, yi) in enumerate(zip(pa, ctxs, ex_y)):
        d0 = decide(p, c, False, False)
        if required_speed(int(d0.cid), c) > required_speed(int(yi), c) + 5:
            picks.append(i)
        if len(picks) == 4:
            break
    if picks:
        fig, axes = plt.subplots(len(picks), 2, figsize=(9, 2.6 * len(picks)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(picks):
            c, yi, p = ctxs[i], int(ex_y[i]), pa[i]
            d0, d1 = decide(p, c, False, False), decide(p, c, True, True)
            axes[r, 0].imshow(adv["PGD"][i]); axes[r, 0].axis("off")
            axes[r, 0].set_title(f"true: {NAMES[yi]}", fontsize=8)
            txt = (f"context: {c.road}, junction={c.junction}, school={c.school}, limit={c.limit}\n"
                   f"network reads: {NAMES[int(np.argmax(p))]} ({p.max():.2f}) -> {d0.speed:.0f} km/h\n"
                   f"{explain(int(np.argmax(p)), c)}\n"
                   f"full system: {'fallback, ' if d1.fallback else ''}"
                   f"{NAMES[d1.cid] if d1.cid is not None else 'safe default'} -> {d1.speed:.0f} km/h "
                   f"(real sign allows {required_speed(yi, c):.0f})")
            axes[r, 1].text(0, 0.5, txt, fontsize=7, va="center", wrap=True)
            axes[r, 1].axis("off")
        fig.suptitle("Attacked images: what the network read and what the full system did", fontsize=10)
        _save(fig, d / "worked_examples.png"); names.append("worked_examples.png")
    return names


def summary(res, out: Path, figs, note: str) -> Path:
    ver = res["verify"]
    vrows = ["| system | combinations checked | P1 acts on impossible sign | P2 over legal max | P3 unsure at junction, not stopping | P4 not most cautious when unsure |",
             "|---|---|---|---|---|---|"]
    for s, r in ver.items():
        v = r["violations"]
        vrows.append(f"| {s} | {r['checked']} | {v['P1']} | {v['P2']} | {v['P3']} | {v['P4']} |")
    ctx_line = (f"Map contexts: **{res.get('context_source', 'simulated')}**"
                + (f" for {res['real_context_images']} of {res['n_eval']} images; the true sign breaks at least one rule "
                   f"in {res['true_sign_breaks_rule']:.1%} of contexts." if res.get("context_source") == "OpenStreetMap" else "."))
    adv = res.get("adv") or {}
    adv_line = (f"Adversarially trained network: clean test accuracy {adv['clean_test_accuracy']:.3f}; attack success "
                + ", ".join(f"{k} {v:.3f}" for k, v in adv["attack_success"].items()) + ".") if adv else ""
    lines = ["# Results: Safe Signs", "", note, "", ctx_line, "", adv_line, "",
             f"Clean test accuracy of the CNN: {res['test_accuracy']:.3f} (temperature {res['temperature']:.2f}). "
             f"Evaluation images: {res['n_eval']}.", "",
             "Attack success on the network (share of images whose top class changed): "
             + ", ".join(f"{k} {v:.3f}" for k, v in res["attack_success"].items()), "",
             "## Decisions under attack", "",
             res["main"].round(4).to_markdown(index=False), "",
             "## Full system vs ablations (paired; negative = full system has fewer unsafe decisions)", "",
             res["comparisons"].round(4).to_markdown(index=False), "",
             "## Choosing the fallback threshold", "",
             res["tau"].round(4).to_markdown(index=False) if res.get("tau") is not None else "", "",
             "## Wrong map facts (context noise)", "",
             res["noise"].round(4).to_markdown(index=False), "",
             "## Rule violations as an attack alarm", "",
             res["detect"].round(4).to_markdown(index=False), "",
             "## Exhaustive verification of the decision layer", "",
             "Every valid context x (one confident sign, or mass split between two signs). Counts are violations.", "",
             *vrows, "", "Counter-examples are listed in verification.json.", "",
             *(["## Rule reliability on the real OpenStreetMap", "", res["osm_reliability"].to_markdown(index=False), ""]
               if res.get("osm_reliability") is not None else []),
             *(["## Soft rule weights", "",
                "Multiplier on a sign's probability when it breaks the rule (0 = hard rule).", "",
                "| rule | weight |", "|---|---|", *[f"| {k} | {v:.4f} |" for k, v in res["rule_weights"].items()], ""]
               if res.get("rule_weights") else []),
             *([f"## Training seeds (standard CNN, {int(res['seeds']['n_seeds'].max())} seeds, same images and contexts)", "",
                res["seeds"].round(4).to_markdown(index=False), ""] if res.get("seeds") is not None else []),
             *_round_four(res),
             "Figures: " + ", ".join(figs), ""]
    p = out / "summary.md"
    p.write_text("\n".join(lines))
    return p


def _md(df):
    return df.round(4).to_markdown(index=False) if df is not None and len(df) else "(not run)"


def _round_four(res) -> list[str]:
    an = res.get("analysis")
    if not an:
        return []
    cert = res.get("certification", {})
    env = res.get("envelope_verify", {})
    erows = ["| system | outputs checked | P5 violations (speed above envelope) | max excess (km/h) |", "|---|---|---|---|"]
    for k, v in env.items():
        erows.append(f"| {k} | {v['checked']} | {v['P5_violations']} | {v['max_excess_kmh']:.0f} |")
    br = an.get("failure_breakdown")
    if br is not None and len(br):
        br = br[br.attack.isin(["PGD", "rule-aware (targeted)"])]
    return [
        "## Baselines (unsafe-decision rate)", "", _md(an.get("baselines")), "",
        "## Baselines (accuracy of the sign acted on)", "", _md(an.get("baselines_accuracy")), "",
        "## Clean vs robust trade-off", "", _md(an.get("tradeoff")), "",
        "## Safety-property violations on the test images", "",
        "P1: acted on a sign the rules call impossible in the context it sees. P2: speed above the legal maximum.", "",
        _md(an.get("runtime_violations")), "",
        "## Decision-level certificate", "",
        f"Certified images (no perception attack of any size can make the hard-logic systems unsafe): "
        f"{cert.get('certified_share', float('nan')):.1%} of {cert.get('images', 0)}.", "",
        f"Worst possible overspeed, mean over images: {cert.get('mean_worst_overspeed_bound_with_rules_kmh', float('nan')):.1f} km/h "
        f"with rules vs {cert.get('mean_worst_overspeed_bound_network_alone_kmh', float('nan')):.1f} km/h for the network alone "
        f"(maximum {cert.get('max_worst_overspeed_bound_with_rules_kmh', float('nan')):.0f} vs "
        f"{cert.get('max_worst_overspeed_bound_network_alone_kmh', float('nan')):.0f}).", "",
        "Empirical check of the certificate on every attack (both counts must be 0):", "",
        _md(an.get("certification_check")), "",
        "## Envelope check (P5) on random network outputs", "", *erows, "",
        "## Stronger attacks", "",
        "Rule-aware adaptive: long targeted PGD towards the fastest rule-consistent unsafe signs, keeping the first "
        "that makes the full system unsafe. Map spoof: one map fact changed to widen the rule envelope.", "",
        _md(an.get("stress")), "", "Which target the adaptive attack needed:", "", _md(an.get("adaptive_targets")), "",
        "## Attack strength curve", "", _md(an.get("eps_sweep")), "",
        "## Failure analysis", "", "Why the full system is unsafe, per attack:", "", _md(an.get("failure_causes")), "",
        "Under the stronger attacks:", "", _md(an.get("stress_failure_causes")), "",
        "When the rules help: images where the network alone was unsafe and the full system safe, credited to the "
        "rule that rejected the network's reading:", "", _md(an.get("rules_that_helped")), "",
        "Unsafe rate by scenario (PGD and rule-aware attacks):", "", _md(br), "",
    ]
