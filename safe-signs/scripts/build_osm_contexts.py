"""Download German traffic signs and their surroundings from OpenStreetMap (needs internet; run
in Colab), and write:
    data/osm_contexts.json         real contexts per GTSRB class, used by run.py if present
    data/osm_rule_reliability.csv  how often each rule is broken on the real map

    python scripts/build_osm_contexts.py
    python scripts/build_osm_contexts.py --areas Berlin München --default-road urban urban
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from ss.osm import overpass_query, parse_overpass, rule_reliability, save_pool  # noqa: E402
from ss.signs import NAMES  # noqa: E402

URL = "https://overpass-api.de/api/interpreter"


def fetch(area: str, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(4):
        r = requests.post(URL, data={"data": overpass_query(area)}, timeout=1000,
                          headers={"User-Agent": "safe-signs-research/1.0"})
        if r.status_code == 200:
            cache.write_text(r.text)
            return r.json()
        print(f"  Overpass returned {r.status_code}, retrying in {30 * (attempt + 1)} s")
        time.sleep(30 * (attempt + 1))
    print(f"  Overpass failed for {area}; skipping it (run again later to add it)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="+",
                    default=["Berlin", "Potsdam", "München", "Hamburg", "Landkreis Märkisch-Oderland"])
    ap.add_argument("--default-road", nargs="+", default=None,
                    help="road type when OSM gives no speed limit, one per area (urban or rural)")
    args = ap.parse_args()
    defaults = args.default_road or ["rural" if a.startswith("Landkreis") else "urban" for a in args.areas]
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    records = []
    for area, dr in zip(args.areas, defaults):
        print(f"downloading {area} ...")
        payload = fetch(area, out / f"osm_v2_{area.replace(' ', '_')}.json")
        if payload is None:
            continue
        recs = parse_overpass(payload, default_road=dr)
        print(f"  {len(recs)} sign records mapped to GTSRB classes")
        records += recs
    if not records:
        raise SystemExit("No signs found. Check the area names (they must match OSM's name tag).")
    pool = save_pool(records, out / "osm_contexts.json")
    rel = pd.DataFrame(rule_reliability(records))
    rel.to_csv(out / "osm_rule_reliability.csv", index=False)
    counts = Counter(r["cid"] for r in records)
    print("\nReal examples per class (classes with fewer than 5 use simulated contexts):")
    for c in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {c:2d} {NAMES[c]:32s} {counts[c]}")
    print("\nRule reliability on the real map:")
    print(rel.to_string(index=False))
    jm = Counter(r["junction_method"] for r in records)
    print(f"junction detection: {dict(jm)}")
    print(f"\nclasses with real contexts: {sum(len(v) >= 5 for v in pool.values())} of 43")


if __name__ == "__main__":
    main()
