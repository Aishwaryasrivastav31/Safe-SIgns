"""Map context for each test image: sampled so that the true sign is plausible there,
optionally corrupted to simulate a wrong map."""

from __future__ import annotations

import numpy as np

from .logic import ALLOWED_SPEEDS, ROADS, Context, consistent


def sample_context(true_cid: int, rng, max_tries: int = 500, pool: dict | None = None) -> Context:
    """A context for an image of class true_cid. With an OpenStreetMap pool, a real context
    for that class is drawn as it is, even if it breaks a rule, because real maps do. Classes
    with too few real examples fall back to the simulated sampler."""
    if pool and true_cid in pool:
        return pool[true_cid][rng.integers(len(pool[true_cid]))]
    for _ in range(max_tries):
        road = ROADS[rng.choice(3, p=[0.5, 0.3, 0.2])]
        junction = road != "motorway" and rng.random() < 0.4
        school = road == "urban" and rng.random() < 0.15
        speeds = sorted(ALLOWED_SPEEDS[road])
        if school:
            speeds = [s for s in speeds if s <= 30]
        limit = None if rng.random() < 0.3 else int(rng.choice(speeds))
        nopass = road != "urban" and rng.random() < 0.3
        c = Context(road, junction, school, limit, nopass)
        if c.valid() and consistent(true_cid, c):
            return c
    raise RuntimeError(f"no consistent context found for class {true_cid}")


def corrupt(ctx: Context, rng) -> Context:
    """A wrong map: change one fact, keeping the context physically possible."""
    for _ in range(100):
        field = rng.integers(5)
        if field == 0:
            new = Context(ROADS[rng.integers(3)], ctx.junction, ctx.school, ctx.limit, ctx.nopass)
        elif field == 1:
            new = Context(ctx.road, not ctx.junction, ctx.school, ctx.limit, ctx.nopass)
        elif field == 2:
            new = Context(ctx.road, ctx.junction, not ctx.school, ctx.limit, ctx.nopass)
        elif field == 3:
            opts = [None] + sorted(ALLOWED_SPEEDS[ctx.road])
            new = Context(ctx.road, ctx.junction, ctx.school, opts[rng.integers(len(opts))], ctx.nopass)
        else:
            new = Context(ctx.road, ctx.junction, ctx.school, ctx.limit, not ctx.nopass)
        if new.valid() and new != ctx:
            return new
    return ctx
