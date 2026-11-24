"""Stronger attacks aimed at the full system, not just the network.

  rule_aware_adaptive  white-box on the network AND aware of the rules and the fallback. For each
                       image it tries the k fastest signs that the rules accept and that would be
                       unsafe, runs a long targeted PGD towards each (so the network becomes
                       confident enough to avoid the fallback), and keeps the first image on which
                       the full system's decision is actually unsafe.
  map_spoof            the attacker also controls one fact of the map context (e.g. a spoofed GPS
                       position or a wrong road-type tag). It picks the single-fact change that
                       widens the rule envelope the most, then runs the rule-aware attack against
                       the spoofed context.
  eps_sweep            attack strength curve: the same attacks at several L-infinity budgets.

These attacks are the "where does the attacker still win" part of the study.
"""

from __future__ import annotations

import numpy as np

from .attacks import pgd
from .logic import ALLOWED_SPEEDS, ROADS, Context, consistency_mask, decide, envelope, is_unsafe, required_speed
from .signs import N_CLASSES


def unsafe_consistent_targets(true_cid: int, true_ctx: Context, seen_ctx: Context, k: int) -> list[int]:
    """Up to k signs, fastest first, that the rules accept in seen_ctx and that would be unsafe."""
    mask = consistency_mask(seen_ctx)
    cands = [c for c in range(N_CLASSES) if mask[c] and c != true_cid
             and is_unsafe(required_speed(c, seen_ctx), true_cid, true_ctx)]
    cands.sort(key=lambda c: -required_speed(c, seen_ctx))
    return cands[:k]


def rule_aware_adaptive(net, x, y, ctx_true, ctx_seen, eps, steps, k, tau, plausible_eps, seed=0,
                        system=(True, True)):
    """Returns (adversarial images, rank of the target that worked, -1 if none, -2 if certified)."""
    adv = x.copy()
    rank_used = np.full(len(x), -1)
    targets = [unsafe_consistent_targets(int(yi), ct, cs, k) for yi, ct, cs in zip(y, ctx_true, ctx_seen)]
    rank_used[[i for i, t in enumerate(targets) if not t]] = -2  # nothing to aim for: certified safe
    best_p = np.full(len(x), -1.0)
    for r in range(k):
        idx = np.array([i for i, t in enumerate(targets) if len(t) > r and rank_used[i] == -1])
        if len(idx) == 0:
            break
        tgt = np.array([targets[i][r] for i in idx])
        cand = pgd(net, x[idx], y[idx], eps, steps, step=eps / 6, target=tgt, seed=seed + r)
        p = net.predict_proba(cand)
        for j, i in enumerate(idx):
            d = decide(p[j], ctx_seen[i], system[0], system[1], tau, plausible_eps)
            if is_unsafe(d.speed, int(y[i]), ctx_true[i]):
                adv[i], rank_used[i] = cand[j], r
            elif p[j, tgt[j]] > best_p[i]:
                adv[i], best_p[i] = cand[j], p[j, tgt[j]]
    return adv.astype(np.float32), rank_used


def single_fact_changes(ctx: Context) -> list[Context]:
    out = []
    for road in ROADS:
        out.append(Context(road, ctx.junction, ctx.school, ctx.limit, ctx.nopass))
    out.append(Context(ctx.road, not ctx.junction, ctx.school, ctx.limit, ctx.nopass))
    out.append(Context(ctx.road, ctx.junction, not ctx.school, ctx.limit, ctx.nopass))
    for lim in [None, *sorted(ALLOWED_SPEEDS[ctx.road])]:
        out.append(Context(ctx.road, ctx.junction, ctx.school, lim, ctx.nopass))
    out.append(Context(ctx.road, ctx.junction, ctx.school, ctx.limit, not ctx.nopass))
    return [c for c in dict.fromkeys(out) if c.valid() and c != ctx]


def worst_spoof(true_cid: int, ctx: Context) -> Context:
    """The single-fact change that lets the car be pushed fastest relative to the real sign."""
    best, best_env = ctx, envelope(ctx)
    for c in single_fact_changes(ctx):
        e = envelope(c)
        if e > best_env:
            best, best_env = c, e
    return best


def eps_sweep(net, x, y, ctxs, eps_list, steps, tau, plausible_eps, seed, systems):
    """Unsafe rate and accuracy of each system for untargeted PGD and the rule-aware targeted attack."""
    from .experiment import most_dangerous_target  # local import avoids a cycle

    rows = []
    tgt = most_dangerous_target(y, ctxs, True)
    for e in eps_list:
        for att, t in (("PGD", None), ("rule-aware (targeted)", tgt)):
            xa = pgd(net, x, y, e, steps, target=t, seed=seed)
            p = net.predict_proba(xa)
            for sname, (logic, fb) in systems.items():
                dec = [decide(pi, c, logic, fb, tau, plausible_eps) for pi, c in zip(p, ctxs)]
                unsafe = np.array([is_unsafe(d.speed, int(yi), c) for d, yi, c in zip(dec, y, ctxs)])
                acc = np.array([d.cid == yi for d, yi in zip(dec, y)])
                rows.append({"eps_255": round(e * 255, 1), "attack": att, "system": sname,
                             "unsafe_rate": float(unsafe.mean()), "accuracy": float(acc.mean()),
                             "fallback_rate": float(np.mean([d.fallback for d in dec]))})
    return rows
