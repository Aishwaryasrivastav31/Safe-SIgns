"""The symbolic layer: a small knowledge base of traffic rules over the map context.

Context facts (what the car knows without looking at the sign):
    road(urban | rural | motorway), junction(yes/no), school_zone(yes/no),
    current_limit(km/h or none), no_passing_active(yes/no)

Integrity constraints say which signs are impossible in a context. Each constraint has a
name and a plain-English reason, so every rejection can be explained. The same table is
exported as a Prolog program (rules.pl) so the rules can be read and checked on their own.

Decision: the system acts on the most probable sign that is consistent with the context.
If that sign is not probable enough, it falls back to the most restrictive behaviour among
the plausible consistent signs (cautious under uncertainty).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signs import DEFAULT_SPEED, MAX_LEGAL, N_CLASSES, SIGNS

ROADS = ("urban", "rural", "motorway")
LIMITS = (None, 20, 30, 50, 60, 70, 80, 100, 120)
ALLOWED_SPEEDS = {"urban": {20, 30, 50, 60}, "rural": {50, 60, 70, 80, 100}, "motorway": {60, 70, 80, 100, 120}}


@dataclass(frozen=True)
class Context:
    road: str
    junction: bool
    school: bool
    limit: int | None
    nopass: bool

    def valid(self) -> bool:
        """Physically possible context (independent of any sign)."""
        if self.road == "motorway" and (self.junction or self.school):
            return False
        if self.school and self.road != "urban":
            return False
        if self.limit is not None and self.limit not in ALLOWED_SPEEDS[self.road]:
            return False
        return True

    def current_speed(self) -> int:
        return self.limit if self.limit is not None else DEFAULT_SPEED[self.road]


# name, reason, test(sign, ctx) -> True if the rule is VIOLATED
RULES = [
    ("speed_fits_road", "this road type does not use that speed limit",
     lambda s, c: s.kind == "speed" and s.speed not in ALLOWED_SPEEDS[c.road]),
    ("school_zone_slow", "school zones only post 20 or 30 km/h",
     lambda s, c: s.kind == "speed" and c.school and s.speed > 30),
    ("junction_signs_need_junction", "stop, yield, right-of-way, signals and roundabout signs only appear at junctions",
     lambda s, c: s.kind in ("stop", "yield", "junction") and not c.junction),
    ("end80_needs_80", "'end of 80' only follows an 80 km/h limit",
     lambda s, c: s.kind == "end80" and c.limit != 80),
    ("end_all_needs_limit", "'end of all limits' needs a limit to end",
     lambda s, c: s.kind == "end_all" and c.limit is None),
    ("end_nopass_needs_nopass", "'end of no passing' needs an active no-passing zone",
     lambda s, c: s.kind == "end_nopass" and not c.nopass),
    ("no_people_on_motorway", "pedestrian, child and cycle crossings do not exist on motorways",
     lambda s, c: s.kind == "people" and c.road == "motorway"),
    ("children_in_town", "children-crossing signs are urban",
     lambda s, c: s.cid == 28 and c.road != "urban"),
    ("animals_outside_town", "wild-animal signs are not used in towns",
     lambda s, c: s.cid == 31 and c.road == "urban"),
    ("no_entry_not_on_motorway", "'no entry' and 'no vehicles' do not stand on a motorway carriageway",
     lambda s, c: s.kind == "no_entry" and c.road == "motorway"),
]


def violations(cid: int, ctx: Context) -> list[str]:
    s = SIGNS[cid]
    return [name for name, _, test in RULES if test(s, ctx)]


def consistent(cid: int, ctx: Context) -> bool:
    return not violations(cid, ctx)


def explain(cid: int, ctx: Context) -> str:
    v = violations(cid, ctx)
    if not v:
        return f"'{SIGNS[cid].name}' is consistent with the context"
    reasons = {n: r for n, r, _ in RULES}
    return f"'{SIGNS[cid].name}' rejected: " + "; ".join(reasons[x] for x in v)


def consistency_mask(ctx: Context) -> np.ndarray:
    return np.array([consistent(c, ctx) for c in range(N_CLASSES)])


# --- what each sign tells the car to do -------------------------------------------------

# Soft rules. weight 0 = hard rule (a violating sign is removed). A weight w in (0, 1) multiplies
# the probability of a sign that breaks that rule, i.e. P(context | sign) relative to a consistent
# sign. Weights are estimated from how often each rule is broken on real OpenStreetMap signs.
RULE_WEIGHT: dict[str, float] = {name: 0.0 for name, _, _ in RULES}


def set_rule_weights(weights: dict[str, float]) -> None:
    for k, v in weights.items():
        if k in RULE_WEIGHT:
            RULE_WEIGHT[k] = float(v)


def weights_from_reliability(rows, alpha: float = 20.0, prior: float = 0.01) -> dict[str, float]:
    """Smoothed violation rate per rule: (violated + alpha*prior) / (n + alpha). A rule with no
    real data stays close to the prior (nearly hard); one that real maps break often gets soft."""
    out = {}
    for r in rows:
        n = r["real_signs_it_applies_to"] or 0
        v = r["violated_on_real_map"] or 0
        out[r["rule"]] = (v + alpha * prior) / (n + alpha)
    return out


def soft_weights(ctx: Context) -> np.ndarray:
    w = np.ones(N_CLASSES)
    for c in range(N_CLASSES):
        for name, _, test in RULES:
            if test(SIGNS[c], ctx):
                w[c] *= RULE_WEIGHT[name]
    return w


def required_speed(cid: int, ctx: Context) -> float:
    """Target speed (km/h) the sign asks for in this context; 0 means stop / do not enter."""
    s = SIGNS[cid]
    cur = ctx.current_speed()
    if s.kind in ("stop", "no_entry"):
        return 0.0
    if s.kind == "yield":
        return 10.0
    if s.kind == "speed":
        return float(s.speed)
    if s.kind in ("end80", "end_all"):
        return float(DEFAULT_SPEED[ctx.road])
    if s.kind == "people":
        return float(min(cur, 30))
    if s.kind == "junction":
        return float(min(cur, 0.6 * cur))
    if s.kind == "warning":
        return float(0.7 * cur)
    return float(cur)  # neutral, no passing, end of no passing


def is_unsafe(decided_speed: float, true_cid: int, true_ctx: Context, tol: float = 5.0) -> bool:
    """Unsafe = going faster than the real sign allows (stop / no entry: any motion)."""
    need = required_speed(true_cid, true_ctx)
    if need == 0.0:
        return decided_speed > 0.0
    return decided_speed > need + tol


@dataclass
class Decision:
    cid: int | None        # sign acted on (None for the generic safe default)
    speed: float
    fallback: bool
    rejected_mass: float   # probability the network put on signs the rules call impossible


def safe_default(ctx: Context) -> float:
    return 0.0 if ctx.junction else float(min(ctx.current_speed(), 30))


def decide(probs: np.ndarray, ctx: Context, use_logic: bool, use_fallback: bool,
           tau: float = 0.6, eps: float = 0.02) -> Decision:
    """The four systems compared in the study:
        logic=False, fallback=False   network alone (argmax)
        logic=False, fallback=True    network + cautious fallback when unsure
        logic=True,  fallback=False   act on the most probable consistent sign
        logic=True,  fallback=True    both (the full neuro-symbolic system)
        logic="soft"                  rules down-weight instead of remove (see RULE_WEIGHT)
    """
    if use_logic == "soft":
        q = probs * soft_weights(ctx)
        rejected = float(probs.sum() - q.sum())
        if q.max() <= 0:
            return Decision(None, safe_default(ctx), True, rejected)
        top = int(np.argmax(q))
        if not use_fallback or q[top] >= tau:
            return Decision(top, required_speed(top, ctx), False, rejected)
        plausible = [c for c in range(N_CLASSES) if q[c] >= eps]
        if not plausible:
            return Decision(None, safe_default(ctx), True, rejected)
        speeds = [required_speed(c, ctx) for c in plausible]
        k = int(np.argmin(speeds))
        return Decision(plausible[k], min(speeds[k], safe_default(ctx) if ctx.junction else speeds[k]), True, rejected)
    mask = consistency_mask(ctx) if use_logic else np.ones(N_CLASSES, bool)
    rejected = float(probs[~mask].sum())
    if not mask.any():
        return Decision(None, safe_default(ctx), True, rejected)
    p = np.where(mask, probs, -1.0)
    top = int(np.argmax(p))
    if not use_fallback or probs[top] >= tau:
        return Decision(top, required_speed(top, ctx), False, rejected)
    plausible = [c for c in range(N_CLASSES) if mask[c] and probs[c] >= eps]
    if not plausible:
        return Decision(None, safe_default(ctx), True, rejected)
    speeds = [required_speed(c, ctx) for c in plausible]
    k = int(np.argmin(speeds))
    return Decision(plausible[k], min(speeds[k], safe_default(ctx) if ctx.junction else speeds[k]), True, rejected)


def envelope(ctx: Context) -> float:
    """The rule envelope: the fastest speed any rule-consistent sign asks for in this context.

    Theorem (decision-level guarantee). For the hard-logic systems (logic only, logic + fallback),
    for EVERY probability vector the network may output, the decided speed is at most envelope(ctx).
    Proof: decide() returns either required_speed(c, ctx) for a sign c that passed the consistency
    mask, the minimum of such speeds (fallback), or safe_default(ctx). Neutral signs are never
    rejected, so envelope(ctx) >= current_speed(ctx) >= safe_default(ctx). Each branch is therefore
    bounded by the maximum over consistent signs. verify.check_envelope() confirms this exhaustively.
    """
    mask = consistency_mask(ctx)
    return max(required_speed(c, ctx) for c in range(N_CLASSES) if mask[c])


def certified_safe(true_cid: int, true_ctx: Context, seen_ctx: Context, tol: float = 5.0) -> bool:
    """True if no perception attack of any size can make a hard-logic system unsafe on this image.

    Follows from the envelope theorem: if even the fastest sign the rules allow in the context the
    system sees is not faster than the real sign allows, no network output can cause an unsafe
    decision. The certificate assumes the map context seen is the one used by the decision layer;
    it says nothing about attacks on the map itself.
    """
    return not is_unsafe(envelope(seen_ctx), true_cid, true_ctx, tol)


def all_contexts() -> list[Context]:
    out = []
    for road in ROADS:
        for j in (False, True):
            for sch in (False, True):
                for lim in LIMITS:
                    for npz in (False, True):
                        c = Context(road, j, sch, lim, npz)
                        if c.valid():
                            out.append(c)
    return out


def to_prolog() -> str:
    """The rule base as a Prolog program, generated from the same table the Python code uses."""
    lines = ["% Safe Signs knowledge base (generated). sign(Id, Name, Kind, Speed).", ""]
    for s in SIGNS:
        sp = s.speed if s.speed is not None else "none"
        lines.append(f"sign({s.cid}, '{s.name}', {s.kind}, {sp}).")
    lines += ["", "allowed_speed(urban, S) :- member(S, [20,30,50,60]).",
              "allowed_speed(rural, S) :- member(S, [50,60,70,80,100]).",
              "allowed_speed(motorway, S) :- member(S, [60,70,80,100,120]).", "",
              "% violates(SignId, ctx(Road, Junction, School, Limit, NoPass), RuleName)",
              "violates(I, ctx(R,_,_,_,_), speed_fits_road) :- sign(I,_,speed,S), \\+ allowed_speed(R,S).",
              "violates(I, ctx(_,_,yes,_,_), school_zone_slow) :- sign(I,_,speed,S), S > 30.",
              "violates(I, ctx(_,no,_,_,_), junction_signs_need_junction) :- sign(I,_,K,_), member(K,[stop,yield,junction]).",
              "violates(I, ctx(_,_,_,L,_), end80_needs_80) :- sign(I,_,end80,_), L \\== 80.",
              "violates(I, ctx(_,_,_,none,_), end_all_needs_limit) :- sign(I,_,end_all,_).",
              "violates(I, ctx(_,_,_,_,no), end_nopass_needs_nopass) :- sign(I,_,end_nopass,_).",
              "violates(I, ctx(motorway,_,_,_,_), no_people_on_motorway) :- sign(I,_,people,_).",
              "violates(28, ctx(R,_,_,_,_), children_in_town) :- R \\== urban.",
              "violates(31, ctx(urban,_,_,_,_), animals_outside_town).",
              "violates(I, ctx(motorway,_,_,_,_), no_entry_not_on_motorway) :- sign(I,_,no_entry,_).",
              "", "consistent(I, Ctx) :- sign(I,_,_,_), \\+ violates(I, Ctx, _).", ""]
    return "\n".join(lines)
