import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ss.attacks import fgsm, patch_masks, pgd  # noqa: E402
from ss.cnn import CNN  # noqa: E402
from ss.contexts import corrupt, sample_context  # noqa: E402
from ss.data import synthetic_signs  # noqa: E402
from ss.experiment import Config, run  # noqa: E402
from ss.logic import RULES, Context, all_contexts, consistent, decide, explain, is_unsafe, to_prolog  # noqa: E402
from ss.verify import check  # noqa: E402


def onehot(c, p=0.99):
    v = np.full(43, (1 - p) / 42)
    v[c] = p
    return v


def test_rules_examples():
    urban = Context("urban", False, False, 50, False)
    assert not consistent(8, urban)          # 120 in town
    assert consistent(2, urban)              # 50 in town
    assert not consistent(14, urban)         # stop without a junction
    assert consistent(14, Context("urban", True, False, 50, False))
    assert not consistent(28, Context("motorway", False, False, None, False))
    assert not consistent(6, urban) and consistent(6, Context("rural", False, False, 80, False))
    assert "town" in explain(31, urban) or "towns" in explain(31, urban)


def test_sampled_contexts_accept_the_true_sign_and_corruption_stays_valid():
    rng = np.random.default_rng(0)
    for c in range(43):
        ctx = sample_context(c, rng)
        assert ctx.valid() and consistent(c, ctx)
        assert corrupt(ctx, rng).valid()


def test_four_systems():
    ctx = Context("urban", True, False, 50, False)
    p = onehot(8)                                         # network sure it is "120"
    assert decide(p, ctx, False, False).speed == 120      # network alone obeys it
    d = decide(p, ctx, True, False)
    assert d.cid != 8 and d.rejected_mass > 0.9           # logic refuses the impossible sign
    p2 = np.zeros(43); p2[2] = p2[14] = 0.5                # torn between 50 and stop
    assert decide(p2, ctx, True, True).speed == 0         # unsure at a junction: stop
    assert is_unsafe(50, 14, ctx) and not is_unsafe(0, 14, ctx)


def test_verification_logic_systems_have_no_violations():
    r = check()
    for name in ("logic only", "logic + fallback"):
        assert all(v == 0 for v in r[name]["violations"].values()), name
    assert r["network alone"]["violations"]["P1"] > 0
    assert len(all_contexts()) > 50


def test_prolog_export_covers_every_rule():
    pl = to_prolog()
    for name, _, _ in RULES:
        assert name in pl
    assert pl.count("sign(") >= 43


def test_cnn_gradients_match_finite_differences():
    net = CNN(n_classes=5, size=8, seed=0, c1=2, c2=3, hidden=4)
    for k in net.p:
        net.p[k] = net.p[k].astype(np.float64)
    net.mean, net.std = np.zeros(3), np.ones(3)
    rng = np.random.default_rng(0)
    x = rng.random((3, 8, 8, 3)); y = np.array([0, 2, 4])

    def loss():
        lg, _ = net._forward(x)
        p = net._softmax(lg)
        return -np.log(p[np.arange(3), y]).mean()

    lg, c = net._forward(x)
    p = net._softmax(lg); p[np.arange(3), y] -= 1
    g, dx = net._backward(p / 3, c, need_input=True)
    for k in ("w1", "b1", "w2", "w4"):
        i = tuple(np.unravel_index(0, net.p[k].shape))
        old = net.p[k][i]
        net.p[k][i] = old + 1e-6; a = loss()
        net.p[k][i] = old - 1e-6; b = loss()
        net.p[k][i] = old
        assert abs((a - b) / 2e-6 - g[k][i]) < 1e-5, k
    i = (1, 2, 5, 0); old = x[i]
    x[i] = old + 1e-6; a = loss(); x[i] = old - 1e-6; b = loss(); x[i] = old
    assert abs((a - b) / 2e-6 - dx[i]) < 1e-5


def test_attacks_respect_budgets():
    x, y = synthetic_signs(3, seed=1)
    net = CNN(seed=0)
    eps = 8 / 255
    for adv in (fgsm(net, x, y, eps), pgd(net, x, y, eps, steps=3)):
        assert np.abs(adv - x).max() <= eps + 1e-6 and adv.min() >= 0 and adv.max() <= 1
    m = patch_masks(len(x), 32, 8)
    adv = pgd(net, x, y, 1.0, steps=3, step=0.2, mask=m)
    assert np.allclose(adv * (1 - m), x * (1 - m))       # pixels outside the sticker unchanged


def test_end_to_end(tmp_path):
    x, y = synthetic_signs(12, seed=2)
    cut = int(0.8 * len(x))
    res = run(x[:cut], y[:cut], x[cut:], y[cut:],
              Config(epochs=1, n_eval=60, pgd_steps=2, patch_steps=2, n_boot=20, extra_seeds=[1], adv_training=False), tmp_path)
    assert set(res["main"]["system"]) == {"network alone", "network + fallback", "logic only", "logic + fallback", "soft logic + fallback"}
    assert (tmp_path / "verification.json").exists()


def test_osm_sign_parsing_and_context_snapping():
    from ss.osm import parse_maxspeed, parse_sign, to_context

    assert parse_sign("DE:206") == [14]
    assert parse_sign("DE:274[50]") == [2] and parse_sign("DE:274-120") == [8]
    assert parse_sign("DE:205;DE:1000-31") == [13]
    assert parse_sign("DE:103-10") == [19] and parse_sign("DE:278-80") == [6]
    assert parse_maxspeed("DE:urban") == 50 and parse_maxspeed("none") is None and parse_maxspeed("30") == 30
    c = to_context({"road": "urban", "junction": True, "school": True, "limit": 40, "nopass": True})
    assert c.valid() and c.limit == 30 and not c.nopass


def test_osm_parser_on_a_fake_overpass_answer():
    from ss.osm import parse_overpass, rule_reliability

    lat, lon = 52.5, 13.4
    d = 0.0001  # about 10 m
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": lat, "lon": lon, "tags": {"traffic_sign": "DE:206"}},
        {"type": "node", "id": 2, "lat": lat + 0.01, "lon": lon, "tags": {"traffic_sign": "DE:274[120]"}},
        {"type": "way", "id": 10, "tags": {"highway": "residential", "name": "A", "maxspeed": "30"},
         "geometry": [{"lat": lat - d, "lon": lon - 3 * d}, {"lat": lat - d, "lon": lon + 3 * d}]},
        {"type": "way", "id": 11, "tags": {"highway": "residential", "name": "B"},
         "geometry": [{"lat": lat - 3 * d, "lon": lon + d}, {"lat": lat + 3 * d, "lon": lon + d}]},
        {"type": "way", "id": 12, "tags": {"highway": "residential", "name": "C", "maxspeed": "50"},
         "geometry": [{"lat": lat + 0.01 - d, "lon": lon - 3 * d}, {"lat": lat + 0.01 - d, "lon": lon + 3 * d}]},
        {"type": "node", "id": 3, "lat": lat + d, "lon": lon, "tags": {"amenity": "school"}},
    ]}
    recs = parse_overpass(payload)
    stop = [r for r in recs if r["cid"] == 14][0]
    fast = [r for r in recs if r["cid"] == 8][0]
    assert stop["junction"] and stop["road"] == "urban" and stop["school"]
    assert not fast["junction"]
    rel = {r["rule"]: r for r in rule_reliability(recs)}
    assert rel["speed_fits_road"]["violated_on_real_map"] == 1    # a 120 sign on a 50 km/h street


def test_osm_pool_is_used_for_sampling(tmp_path):
    from ss.osm import load_pool, save_pool

    recs = [{"cid": 14, "road": "rural", "junction": True, "school": False, "limit": 70, "nopass": False}] * 6
    save_pool(recs, tmp_path / "p.json")
    pool = load_pool(tmp_path / "p.json")
    ctx = sample_context(14, np.random.default_rng(0), pool=pool)
    assert ctx.road == "rural" and ctx.junction


def test_adversarial_training_runs():
    x, y = synthetic_signs(4, seed=3)
    net = CNN(seed=0).fit(x[:120], y[:120], x[120:], y[120:], epochs=1, adv_eps=8 / 255, adv_steps=1, log=lambda *_: None)
    assert np.isfinite(net.logits(x[:5])).all()


def test_soft_logic_equals_hard_when_weights_are_zero_and_softens_otherwise():
    from ss.logic import RULE_WEIGHT, Context, decide, set_rule_weights

    old = dict(RULE_WEIGHT)
    try:
        set_rule_weights({k: 0.0 for k in RULE_WEIGHT})
        rng = np.random.default_rng(0)
        ctx = Context("rural", False, False, 70, False)
        for _ in range(50):
            p = rng.dirichlet(np.ones(43) * 0.3)
            a, b = decide(p, ctx, True, True), decide(p, ctx, "soft", True)
            assert a.cid == b.cid and a.speed == b.speed
        p = np.full(43, 0.001); p[14] = 1 - 0.042   # confident stop sign, no junction on the map
        assert decide(p, ctx, True, False).cid != 14                 # hard rule removes it
        set_rule_weights({"junction_signs_need_junction": 0.8})
        assert decide(p, ctx, "soft", False).cid == 14               # soft rule keeps it
    finally:
        set_rule_weights(old)


def test_weights_from_reliability():
    from ss.logic import weights_from_reliability

    w = weights_from_reliability([{"rule": "a", "real_signs_it_applies_to": 126, "violated_on_real_map": 0},
                                  {"rule": "b", "real_signs_it_applies_to": 52, "violated_on_real_map": 11},
                                  {"rule": "c", "real_signs_it_applies_to": 0, "violated_on_real_map": 0}])
    assert w["a"] < 0.005 and 0.1 < w["b"] < 0.2 and abs(w["c"] - 0.01) < 1e-9


def test_osm_junction_from_shared_nodes():
    from ss.osm import parse_overpass

    lat, lon, d = 52.5, 13.4, 0.0001
    def way(i, name, pts, nodes):
        return {"type": "way", "id": i, "tags": {"highway": "residential", "name": name},
                "nodes": nodes, "geometry": [{"lat": a, "lon": b} for a, b in pts]}
    # sign 1: two different streets meet at node 100 next to it -> junction
    # sign 2: one street split into two ways sharing node 200 -> not a junction
    lat2 = lat + 0.01
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": lat, "lon": lon, "tags": {"traffic_sign": "DE:206"}},
        {"type": "node", "id": 2, "lat": lat2, "lon": lon, "tags": {"traffic_sign": "DE:206"}},
        way(10, "A", [(lat - d, lon - 3 * d), (lat - d, lon)], [99, 100]),
        way(11, "B", [(lat - d, lon), (lat - 3 * d, lon)], [100, 101]),
        way(12, "C", [(lat2 - d, lon - 3 * d), (lat2 - d, lon)], [199, 200]),
        way(13, "C", [(lat2 - d, lon), (lat2 - d, lon + 3 * d)], [200, 201]),
    ]}
    recs = {r["node"]: r for r in parse_overpass(payload)}
    assert recs[1]["junction"] and recs[1]["junction_method"] == "topology"
    assert not recs[2]["junction"]


# ---- round four -----------------------------------------------------------------------------

def test_envelope_bounds_every_hard_logic_decision():
    from ss.logic import MAX_LEGAL, envelope

    rng = np.random.default_rng(5)
    for ctx in all_contexts():
        env = envelope(ctx)
        assert env <= MAX_LEGAL[ctx.road]
        for _ in range(40):
            p = rng.dirichlet(np.full(43, rng.choice([1.0, 0.05])))
            for logic, fb in ((True, False), (True, True)):
                assert decide(p, ctx, logic, fb).speed <= env + 1e-9


def test_certificate_means_no_attack_can_win():
    from ss.adaptive import unsafe_consistent_targets
    from ss.logic import certified_safe

    for ctx in all_contexts()[:30]:
        for c in range(43):
            if not consistent(c, ctx):
                continue
            # certified exactly when there is no rule-consistent sign that would be unsafe
            assert certified_safe(c, ctx, ctx) == (len(unsafe_consistent_targets(c, ctx, ctx, 43)) == 0)


def test_envelope_check_finds_no_violations_for_logic():
    from ss.verify import check_envelope

    r = check_envelope(n_random=20)
    assert r["logic only"]["P5_violations"] == 0 and r["logic + fallback"]["P5_violations"] == 0
    assert r["network alone"]["P5_violations"] > 0


def test_worst_spoof_widens_or_keeps_the_envelope_and_stays_valid():
    from ss.adaptive import single_fact_changes, worst_spoof
    from ss.logic import envelope

    for ctx in all_contexts():
        s = worst_spoof(0, ctx)
        assert s.valid() and envelope(s) >= envelope(ctx)
        assert all(c.valid() and c != ctx for c in single_fact_changes(ctx))


def test_adaptive_attack_respects_budget_and_reports_ranks():
    from ss.adaptive import rule_aware_adaptive

    x, y = synthetic_signs(3, seed=4)
    net = CNN(seed=0, c1=4, c2=8, hidden=16)
    rng = np.random.default_rng(0)
    ctxs = [sample_context(int(c), rng) for c in y]
    adv, rank = rule_aware_adaptive(net, x, y, ctxs, ctxs, 8 / 255, 3, 2, 0.6, 0.02)
    assert np.abs(adv - x).max() <= 8 / 255 + 1e-6
    assert set(np.unique(rank)) <= {-2, -1, 0, 1}


def test_failure_analysis_categories():
    import pandas as pd

    from ss.analysis import failure_analysis

    ctx = Context("urban", True, False, 50, False)
    wrong = Context("rural", False, False, None, False)
    y = np.array([14, 14])
    base = {"unsafe": np.array([True, True]), "fallback": np.array([False, False])}
    per = {"a": {"logic + fallback": base, "network alone": base}}
    probs = {"a": np.stack([onehot(2), onehot(2)])}
    causes, _, _ = failure_analysis(per, probs, y, [ctx, ctx], [ctx, wrong])
    c = causes.set_index("cause").images
    assert c["inside the rule envelope"] == 1 and c["map wrong"] == 1
    assert isinstance(causes, pd.DataFrame)
