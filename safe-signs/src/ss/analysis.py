"""Analysis added in round four: baselines, runtime property violations, the clean/robust
trade-off, decision-level certification, and failure cases (when rules help, when they fail,
when the attacker still wins)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .logic import RULES, envelope, is_unsafe, required_speed, violations
from .signs import N_CLASSES, SIGNS

BASELINES = [
    ("Vanilla CNN", "standard", "network alone"),
    ("Adversarially trained CNN", "adversarially trained", "network alone"),
    ("CNN + confidence threshold", "standard", "network + fallback"),
    ("Full system (CNN + rules + fallback)", "standard", "logic + fallback"),
    ("Adv. trained CNN + rules + fallback", "adversarially trained", "logic + fallback"),
]


def baselines_table(main: pd.DataFrame, metric: str = "unsafe_rate") -> pd.DataFrame:
    rows = []
    for att in dict.fromkeys(main.attack):
        r = {"attack": att}
        for label, model, system in BASELINES:
            m = main[(main.attack == att) & (main.model == model) & (main.system == system)]
            if len(m):
                r[label] = float(m[metric].iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)


def tradeoff_table(main: pd.DataFrame) -> pd.DataFrame:
    """Clean accuracy vs robust accuracy (PGD) and the matching unsafe rates, per model and system."""
    rows = []
    for (model, system), d in main.groupby(["model", "system"], sort=False):
        g = d.set_index("attack")
        rows.append({"model": model, "system": system,
                     "clean_accuracy": g.loc["clean", "accuracy"], "pgd_accuracy": g.loc["PGD", "accuracy"],
                     "clean_unsafe": g.loc["clean", "unsafe_rate"], "pgd_unsafe": g.loc["PGD", "unsafe_rate"],
                     "worst_attack_unsafe": d[d.attack != "clean"].unsafe_rate.max(),
                     "clean_fallback": g.loc["clean", "fallback_rate"]})
    return pd.DataFrame(rows)


def runtime_violations(per_image_by_model: dict) -> pd.DataFrame:
    """How often each system breaks P1 (acts on a sign the rules call impossible in the context it
    sees) and P2 (decides a speed above the legal maximum) on the real test images."""
    rows = []
    for model, per_image in per_image_by_model.items():
        for att, res in per_image.items():
            for sysname, r in res.items():
                rows.append({"model": model, "attack": att, "system": sysname,
                             "P1_acted_on_impossible_sign": int(r["acted_impossible"].sum()),
                             "P2_over_legal_max": int(r["over_legal"].sum()), "images": len(r["unsafe"])})
    return pd.DataFrame(rows)


def certification(y, ctx_true, ctx_seen, per_image: dict, tol: float = 5.0) -> tuple[dict, pd.DataFrame]:
    """Decision-level certificate. An image is certified if even the fastest rule-consistent sign is
    not unsafe, so NO perception attack can make a hard-logic system unsafe on it. Also the worst
    possible overspeed (the guaranteed bound) with and without rules."""
    need = np.array([required_speed(int(yi), ct) for yi, ct in zip(y, ctx_true)])
    env = np.array([envelope(cs) for cs in ctx_seen])
    worst_net = np.array([max(required_speed(c, cs) for c in range(N_CLASSES)) for cs in ctx_seen])
    cert = np.array([not is_unsafe(e, int(yi), ct, tol) for e, yi, ct in zip(env, y, ctx_true)])
    bound_logic = np.maximum(env - need, 0)
    bound_net = np.maximum(worst_net - need, 0)
    summ = {"images": int(len(y)), "certified_share": float(cert.mean()),
            "mean_worst_overspeed_bound_with_rules_kmh": float(bound_logic.mean()),
            "mean_worst_overspeed_bound_network_alone_kmh": float(bound_net.mean()),
            "max_worst_overspeed_bound_with_rules_kmh": float(bound_logic.max()),
            "max_worst_overspeed_bound_network_alone_kmh": float(bound_net.max())}
    rows = []
    for att, res in per_image.items():
        for sysname in ("logic only", "logic + fallback"):
            r = res[sysname]
            over = r["speed"] - need
            rows.append({"attack": att, "system": sysname,
                         "unsafe_on_certified_images": int((r["unsafe"] & cert).sum()),
                         "overspeed_above_bound": int((over > bound_logic + 1e-9).sum()),
                         "unsafe_share_of_uncertified": float(r["unsafe"][~cert].mean()) if (~cert).any() else 0.0})
    return summ, pd.DataFrame(rows)


def _cause(r, i, yi, cs, ct):
    """Why the full system made an unsafe decision on image i."""
    if cs != ct:
        return "map wrong"
    if violations(int(yi), cs):
        return "true sign breaks a rule"
    if r["fallback"][i]:
        return "fallback still too fast"
    return "inside the rule envelope"


def failure_analysis(per_image: dict, probs: dict, y, ctx_true, ctx_seen, system: str = "logic + fallback"):
    """Returns (causes, rules_that_helped, breakdown) tables.

    causes          per attack: share of images where the full system was unsafe, split by cause
    rules_helped    per attack and rule: images where the network alone was unsafe and the full
                    system was safe, credited to the rule that rejected the network's reading
    breakdown       unsafe rate of network alone vs full system by road type, junction and sign kind
    """
    cause_rows, help_rows, br_rows = [], [], []
    n = len(y)
    for att, res in per_image.items():
        seen = ctx_seen[att] if isinstance(ctx_seen, dict) else ctx_seen
        r, r0 = res[system], res["network alone"]
        counts = {}
        for i in np.where(r["unsafe"])[0]:
            c = _cause(r, i, y[i], seen[i], ctx_true[i])
            counts[c] = counts.get(c, 0) + 1
        for c in ("inside the rule envelope", "fallback still too fast", "true sign breaks a rule", "map wrong"):
            cause_rows.append({"attack": att, "cause": c, "images": counts.get(c, 0),
                               "share_of_all_images": counts.get(c, 0) / n,
                               "share_of_unsafe": counts.get(c, 0) / max(int(r["unsafe"].sum()), 1)})
        saved = np.where(r0["unsafe"] & ~r["unsafe"])[0]
        credit = {name: 0 for name, _, _ in RULES}
        credit["fallback only (reading was rule-consistent)"] = 0
        for i in saved:
            v = violations(int(np.argmax(probs[att][i])), seen[i])
            if v:
                for name in v:
                    credit[name] += 1
            else:
                credit["fallback only (reading was rule-consistent)"] += 1
        for name, k in credit.items():
            if k:
                help_rows.append({"attack": att, "rule": name, "images_saved": k, "share_of_saved": k / max(len(saved), 1)})
        groups = {
            "road": [c.road for c in ctx_true],
            "junction": ["junction" if c.junction else "no junction" for c in ctx_true],
            "sign kind": [SIGNS[int(yi)].kind for yi in y],
        }
        for gname, vals in groups.items():
            vals = np.array(vals)
            for v in sorted(set(vals)):
                m = vals == v
                br_rows.append({"attack": att, "by": gname, "value": v, "images": int(m.sum()),
                                "network_alone_unsafe": float(r0["unsafe"][m].mean()),
                                "full_system_unsafe": float(r["unsafe"][m].mean())})
    return pd.DataFrame(cause_rows), pd.DataFrame(help_rows), pd.DataFrame(br_rows)
