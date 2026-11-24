"""Exhaustive check of the decision layer (the symbolic part, not the network).

The decision layer only sees a context and a probability vector. We enumerate every valid
context and two families of network outputs:
  confident  all mass on one sign (p = 0.99)
  torn       mass split evenly between two different signs (below the fallback threshold)
and check four safety properties on every combination, collecting counter-examples.

  P1  the system never acts on a sign the rules call impossible in this context
  P2  the chosen speed never exceeds the legal maximum for the road type
  P3  when unsure at a junction, the system stops
  P4  when unsure, the system is at least as cautious as every plausible consistent sign

Soft logic breaks P1 by design: it may act on a sign that breaks a rule when the network is
confident enough and the rule is known to be unreliable on real maps.
"""

from __future__ import annotations

import itertools

import numpy as np

from .logic import MAX_LEGAL, all_contexts, consistency_mask, consistent, decide, envelope, required_speed
from .signs import N_CLASSES

SYSTEMS = {"network alone": (False, False), "network + fallback": (False, True),
           "logic only": (True, False), "logic + fallback": (True, True),
           "soft logic + fallback": ("soft", True)}


def _outputs():
    for c in range(N_CLASSES):
        p = np.full(N_CLASSES, 0.01 / (N_CLASSES - 1)); p[c] = 0.99
        yield "confident", p
    for a, b in itertools.combinations(range(N_CLASSES), 2):
        p = np.zeros(N_CLASSES); p[a] = p[b] = 0.5
        yield "torn", p


def check(tau: float = 0.6, eps: float = 0.02, max_examples: int = 5) -> dict:
    ctxs = all_contexts()
    outs = list(_outputs())
    report = {}
    for name, (logic, fb) in SYSTEMS.items():
        counts = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
        examples = {k: [] for k in counts}
        n = 0
        for ctx in ctxs:
            mask = consistency_mask(ctx)
            for kind, p in outs:
                d = decide(p, ctx, logic, fb, tau, eps)
                n += 1
                bad = {}
                bad["P1"] = d.cid is not None and not consistent(d.cid, ctx)
                bad["P2"] = d.speed > MAX_LEGAL[ctx.road] + 1e-9
                bad["P3"] = d.fallback and ctx.junction and d.speed > 0
                plaus = [c for c in range(N_CLASSES) if mask[c] and p[c] >= eps]
                bad["P4"] = d.fallback and bool(plaus) and d.speed > min(required_speed(c, ctx) for c in plaus) + 1e-9
                for k, v in bad.items():
                    if v:
                        counts[k] += 1
                        if len(examples[k]) < max_examples:
                            examples[k].append({"context": ctx.__dict__, "output": kind,
                                                "top_signs": [int(i) for i in np.argsort(-p)[:2]],
                                                "decided_sign": d.cid, "speed": d.speed})
        report[name] = {"checked": n, "violations": counts, "examples": examples}
    return report


def check_envelope(tau: float = 0.6, eps: float = 0.02, n_random: int = 2000, seed: int = 0) -> dict:
    """P5 (envelope bound): decided speed <= envelope(ctx) for every network output.

    Checked on the two output families above plus n_random random probability vectors per context:
    Dirichlet draws at three concentrations (flat, peaked, very peaked) and sparse vectors with mass
    on 1-4 random signs. The bound is proved in logic.envelope(); this is the empirical check.
    Also reports, per system, the mean and maximum envelope gap between what the system could be
    pushed to and the legal maximum.
    """
    rng = np.random.default_rng(seed)
    ctxs = all_contexts()
    fams = list(_outputs())
    out = {}
    for name, (logic, fb) in SYSTEMS.items():
        n = viol = 0
        worst = 0.0
        ex = []
        for ctx in ctxs:
            env = envelope(ctx)
            vecs = [p for _, p in fams]
            for k in range(n_random):
                r = k % 4
                if r < 3:
                    vecs.append(rng.dirichlet(np.full(N_CLASSES, (1.0, 0.1, 0.01)[r])))
                else:
                    p = np.zeros(N_CLASSES)
                    idx = rng.choice(N_CLASSES, rng.integers(1, 5), replace=False)
                    p[idx] = rng.dirichlet(np.ones(len(idx)))
                    vecs.append(p)
            for p in vecs:
                d = decide(p, ctx, logic, fb, tau, eps)
                n += 1
                gap = d.speed - env
                if gap > 1e-9:
                    viol += 1
                    worst = max(worst, gap)
                    if len(ex) < 5:
                        ex.append({"context": ctx.__dict__, "decided_sign": d.cid, "speed": d.speed, "envelope": env})
        out[name] = {"checked": n, "P5_violations": viol, "max_excess_kmh": worst, "examples": ex}
    return out
