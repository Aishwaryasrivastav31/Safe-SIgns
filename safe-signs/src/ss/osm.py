"""Real map context from OpenStreetMap.

GTSRB images carry no location, so an image cannot be tied to its own street. What OSM can
give is the real distribution of places where each kind of German sign stands: which road
type, whether a junction is close, the limit in force, a school nearby, a no-passing stretch.
Each test image then gets a context drawn from real places for its class, instead of from
hand-set probabilities. The same data shows how often the rule base is broken on real maps.

Signs are OSM nodes tagged traffic_sign=DE:xxx. Each sign is matched to the nearest highway
within 30 m. A junction is taken from the road network itself: an OSM node shared by at least two
different roads (different names, or unnamed different ways), or a roundabout, within 30 m of the
sign. Older downloads without node lists fall back to "two differently named roads within 30 m".
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .logic import ALLOWED_SPEEDS, RULES, Context, all_contexts
from .signs import N_CLASSES, SIGNS

# German StVO sign numbers -> GTSRB class ids
DE_BASE = {"206": 14, "205": 13, "306": 12, "301": 11, "267": 17, "250": 15, "253": 16, "276": 9, "277": 10,
           "280": 41, "281": 42, "282": 32, "101": 18, "105": 21, "112": 22, "114": 23, "121": 24, "123": 25,
           "131": 26, "133": 27, "136": 28, "138": 29, "142": 31, "215": 40}
DE_SUB = {"103-10": 19, "103-20": 20, "101-51": 30, "209-20": 33, "209-10": 34, "209-30": 35, "214-20": 36,
          "214-10": 37, "222-20": 38, "222-10": 39, "222": 38}
SPEED_CLASS = {20: 0, 30: 1, 50: 2, 60: 3, 70: 4, 80: 5, 100: 7, 120: 8}
ROAD_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "road"}


def parse_sign(tag: str) -> list[int]:
    """'DE:274[50]', 'DE:274-50', 'DE:206', 'DE:205;DE:1000-31' -> GTSRB class ids found."""
    out = []
    for part in re.split(r"[;,]", tag or ""):
        part = part.strip()
        m = re.match(r"(?:DE:)?(\d{3})(?:[-\[](\d+)\]?)?", part)
        if not m:
            continue
        base, sub = m.group(1), m.group(2)
        if base == "274" and sub and int(sub) in SPEED_CLASS:
            out.append(SPEED_CLASS[int(sub)])
        elif base == "278" and sub == "80":
            out.append(6)
        elif sub and f"{base}-{sub}" in DE_SUB:
            out.append(DE_SUB[f"{base}-{sub}"])
        elif base in DE_SUB:
            out.append(DE_SUB[base])
        elif base in DE_BASE:
            out.append(DE_BASE[base])
    return out


def overpass_query(area: str) -> str:
    return f"""[out:json][timeout:900];
area["name"="{area}"]["boundary"="administrative"]->.a;
node(area.a)["traffic_sign"~"^DE:"]->.signs;
.signs out body;
way(around.signs:35)["highway"];
out body geom;
nwr(around.signs:150)["amenity"="school"];
out center tags;"""


def parse_maxspeed(v: str | None):
    if not v:
        return None
    v = v.strip().lower()
    if v in ("de:urban",):
        return 50
    if v in ("de:rural",):
        return 100
    if v in ("none", "de:motorway", "signals", "walk"):
        return None
    m = re.match(r"(\d+)", v)
    return int(m.group(1)) if m else None


def _road_type(tags: dict, default_road: str) -> str:
    hw = tags.get("highway", "")
    if hw in ("motorway", "motorway_link", "trunk", "trunk_link"):
        return "motorway"
    if hw in ("residential", "living_street", "service", "pedestrian"):
        return "urban"
    ms = parse_maxspeed(tags.get("maxspeed"))
    if ms is not None:
        return "urban" if ms <= 60 else "rural"
    return default_road


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.where(L == 0, 1, L), 0, 1)
    return np.hypot(ax + t * dx - px, ay + t * dy - py)


def parse_overpass(payload: dict, default_road: str = "urban", radius_m: float = 30.0,
                   school_m: float = 150.0) -> list[dict]:
    """One record per (sign node, GTSRB class) with the raw context found around it."""
    els = payload.get("elements", [])
    signs = [e for e in els if e["type"] == "node" and "traffic_sign" in e.get("tags", {})]
    ways = [e for e in els if e["type"] == "way" and "geometry" in e and e.get("tags", {}).get("highway")]
    schools = []
    for e in els:
        if e.get("tags", {}).get("amenity") == "school":
            c = e.get("center") or ({"lat": e["lat"], "lon": e["lon"]} if "lat" in e else None)
            if c:
                schools.append((c["lat"], c["lon"]))
    if not signs:
        return []
    lat0 = np.mean([s["lat"] for s in signs])
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540

    def xy(lat, lon):
        return lon * kx, lat * ky

    segs, seg_way = [], []
    for wi, w in enumerate(ways):
        g = w["geometry"]
        for a, b in zip(g[:-1], g[1:]):
            ax, ay = xy(a["lat"], a["lon"]); bx, by = xy(b["lat"], b["lon"])
            segs.append((ax, ay, bx, by)); seg_way.append(wi)
    segs = np.array(segs) if segs else np.zeros((0, 4))
    seg_way = np.array(seg_way, int)
    cell = 60.0
    grid = defaultdict(list)
    for i, (ax, ay, bx, by) in enumerate(segs):
        for gx in range(int(min(ax, bx) // cell), int(max(ax, bx) // cell) + 1):
            for gy in range(int(min(ay, by) // cell), int(max(ay, by) // cell) + 1):
                grid[(gx, gy)].append(i)
    school_xy = np.array([xy(a, b) for a, b in schools]) if schools else np.zeros((0, 2))

    # Junction nodes from topology: a node shared by two different roads.
    has_topology = any("nodes" in w for w in ways)
    node_roads, node_xy = defaultdict(set), {}
    for wi, w in enumerate(ways):
        if w["tags"].get("highway") not in ROAD_CLASSES or "nodes" not in w:
            continue
        ident = w["tags"].get("name") or w["tags"].get("ref") or f"#{wi}"
        for nid, g in zip(w["nodes"], w["geometry"]):
            node_roads[nid].add(ident)
            node_xy[nid] = xy(g["lat"], g["lon"])
        if w["tags"].get("junction") in ("roundabout", "circular"):
            for nid in w["nodes"]:
                node_roads[nid].add(f"roundabout#{wi}")
                node_roads[nid].add(f"roundabout2#{wi}")
    jxy = np.array([node_xy[n] for n, r in node_roads.items() if len(r) >= 2]) if node_roads else np.zeros((0, 2))

    records = []
    for s in signs:
        cids = parse_sign(s["tags"]["traffic_sign"])
        if not cids:
            continue
        px, py = xy(s["lat"], s["lon"])
        gx, gy = int(px // cell), int(py // cell)
        cand = sorted({i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in grid.get((gx + dx, gy + dy), [])})
        if not cand:
            continue
        cand = np.array(cand)
        d = _seg_dist(px, py, *segs[cand].T)
        near = d <= radius_m
        if not near.any():
            continue
        nearest_way = ways[seg_way[cand[np.argmin(d)]]]
        road_ways = {seg_way[c] for c, n in zip(cand, near) if n and ways[seg_way[c]]["tags"].get("highway") in ROAD_CLASSES}
        names = {ways[w]["tags"].get("name") or f"#{w}" for w in road_ways}
        tags = nearest_way["tags"]
        if has_topology:
            junction = bool(len(jxy)) and bool((np.hypot(*(jxy - [px, py]).T) <= radius_m).any())
        else:
            junction = len(names) >= 2
        rec = {"node": s["id"], "sign_tag": s["tags"]["traffic_sign"],
               "road": _road_type(tags, default_road), "junction": junction,
               "junction_method": "topology" if has_topology else "names",
               "school": bool(len(school_xy)) and bool((np.hypot(*(school_xy - [px, py]).T) <= school_m).any()),
               "raw_maxspeed": tags.get("maxspeed"), "limit": parse_maxspeed(tags.get("maxspeed")),
               "nopass": tags.get("overtaking") == "no"}
        for c in cids:
            records.append({**rec, "cid": c})
    return records


def to_context(rec: dict) -> Context:
    """Snap a raw OSM context to one the rule base can represent."""
    road = rec["road"]
    junction = bool(rec["junction"]) and road != "motorway"
    school = bool(rec["school"]) and road == "urban"
    lim = rec["limit"]
    if lim is not None and lim not in ALLOWED_SPEEDS[road]:
        allowed = sorted(ALLOWED_SPEEDS[road])
        lim = min(allowed, key=lambda a: abs(a - lim))
    if school and lim is not None and lim > 30:
        lim = 30
    return Context(road, junction, school, lim, bool(rec["nopass"]) and road != "urban")


def rule_reliability(records: list[dict]) -> list[dict]:
    """For each rule: how many real signs it could apply to, and how often real maps break it."""
    can_fire = {name: {c for c in range(N_CLASSES) if any(t(SIGNS[c], ctx) for ctx in all_contexts())}
                for name, _, t in RULES}
    rows = []
    for name, reason, test in RULES:
        rel = [r for r in records if r["cid"] in can_fire[name]]
        broken = [r for r in rel if test(SIGNS[r["cid"]], to_context(r))]
        rows.append({"rule": name, "reason": reason, "real_signs_it_applies_to": len(rel),
                     "violated_on_real_map": len(broken),
                     "violation_share": round(len(broken) / len(rel), 4) if rel else None})
    return rows


def save_pool(records: list[dict], path: str | Path) -> dict:
    pool = defaultdict(list)
    for r in records:
        c = to_context(r)
        if c.valid():
            pool[int(r["cid"])].append([c.road, c.junction, c.school, c.limit, c.nopass])
    Path(path).write_text(json.dumps({"pool": pool, "counts": Counter(r["cid"] for r in records)}, default=int))
    return pool


def load_pool(path: str | Path, min_examples: int = 5) -> dict[int, list[Context]]:
    d = json.loads(Path(path).read_text())
    return {int(k): [Context(*v) for v in vs] for k, vs in d["pool"].items() if len(vs) >= min_examples}
