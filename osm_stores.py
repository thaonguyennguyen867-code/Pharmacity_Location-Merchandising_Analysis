"""
osm_stores.py - download REAL Pharmacity locations (coordinates, and opening hours where mapped)
from OpenStreetMap via the Overpass API and turn them into the app's store CSV.

Usage (needs internet):
    python osm_stores.py                              # -> data/stores_hcmc_osm.csv
    python osm_stores.py --bbox 10.62,106.52,10.95,106.90 --out data/my_stores.csv

If the Overpass servers refuse your connection (common from cloud hosts), do it by hand:
    python osm_stores.py --print-query                # copy the query into https://overpass-turbo.eu and run it
    #   Export -> "raw OSM data" -> download as .json, then:
    python osm_stores.py --from-json export.json      # -> data/stores_hcmc_osm.csv

For a deployed app (Streamlit Cloud etc.) generate the CSV once and COMMIT it to the repository,
so the deployed app never needs to call Overpass at runtime.

What comes from OpenStreetMap (real):   name, lat/lng, opening_hours (if the mapper added it), address tags.
What is NOT in OpenStreetMap (assumed): car parking, one-way street, 24/7 status when opening_hours is missing.
Those columns are filled with clearly-marked defaults; edit the CSV (or survey) for the ones you care about.
Columns added for transparency: `hours_source` ("osm" or "assumed") and `osm_url`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OVERPASS_URLS = [  # public mirrors, tried in order
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# (south, west, north, east) - HCMC urban core. Widen it if you want the outer districts too.
DEFAULT_BBOX = (10.62, 106.52, 10.95, 106.90)
ASSUMED_HOURS = ("07:30", "22:00")  # used only when OSM has no opening_hours for a store
USER_AGENT = "pharmacity-hcmc-academic-demo/1.0 (student project)"

_RANGE = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")


def build_query(bbox=DEFAULT_BBOX, timeout: int = 40) -> str:
    """Each statement starts from an indexed tag (amenity/healthcare/shop), so it is fast even on busy servers."""
    s, w, n, e = bbox
    box = f"({s},{w},{n},{e})"
    stmts = [
        f'  nwr["{k}"="{v}"]["{key}"~"pharmacity",i]{box};'
        for k, v in (("amenity", "pharmacy"), ("healthcare", "pharmacy"), ("shop", "chemist"))
        for key in ("name", "brand", "operator")
    ]
    return f"[out:json][timeout:{timeout}];\n(\n" + "\n".join(stmts) + "\n);\nout center tags;"


def _short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:80]}"


def fetch_elements(bbox=DEFAULT_BBOX, urls=None, timeout: int = 40, session=None, deadline: float = 120.0) -> list[dict]:
    """POST the query to Overpass mirrors until one answers.

    `deadline` is the total time budget in seconds (no new mirror is tried after it is used up).
    Raises RuntimeError("Overpass API unreachable -> ...") if every mirror fails.
    """
    sess = session or requests
    t0 = time.monotonic()
    errors = []
    for url in urls or OVERPASS_URLS:
        if errors and time.monotonic() - t0 > deadline:
            errors.append("stopped: time budget used up")
            break
        host = url.split("/")[2] if "//" in url else url
        try:
            r = sess.post(url, data={"data": build_query(bbox, timeout)}, headers={"User-Agent": USER_AGENT},
                          timeout=(10, timeout + 10))
            r.raise_for_status()
            return r.json().get("elements", [])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{host} {_short(exc)}")
    raise RuntimeError("Overpass API unreachable -> " + " | ".join(errors))


def elements_from_json_file(path: str | Path) -> list[dict]:
    """Read a raw Overpass JSON export (e.g. saved from overpass-turbo.eu: Export -> raw OSM data)."""
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(js, dict) and "elements" in js:
        return js["elements"]
    raise ValueError("This file has no 'elements' list. In overpass-turbo choose Export -> 'raw OSM data' (JSON), "
                     "not GeoJSON.")


def parse_opening_hours(value: str | None) -> tuple[str, str, bool, str]:
    """OSM opening_hours -> (open_time, close_time, is_24h, source).

    Handles the common simple patterns: "24/7", "Mo-Su 07:00-22:00", "06:30-01:00" (overnight),
    split shifts "08:00-12:00,14:00-20:00" (-> earliest open to latest close).
    Anything it cannot read falls back to ASSUMED_HOURS with source="assumed".
    """
    if not value or not str(value).strip():
        return (*ASSUMED_HOURS, False, "assumed")
    text = str(value).strip()
    if re.fullmatch(r"24\s*/\s*7", text, flags=re.I):
        return "00:00", "24:00", True, "osm"
    ranges = []
    for h1, m1, h2, m2 in _RANGE.findall(text):
        o, c = int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)
        if o <= 1440 and c <= 1440 and int(m1) < 60 and int(m2) < 60:
            ranges.append((o, c))
    if not ranges:
        return (*ASSUMED_HOURS, False, "assumed")
    if any(o == 0 and c == 1440 for o, c in ranges):
        return "00:00", "24:00", True, "osm"
    overnight = [r for r in ranges if r[1] <= r[0]]
    if overnight:
        o, c = overnight[0]
    else:
        o, c = min(r[0] for r in ranges), max(r[1] for r in ranges)
    fmt = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    return fmt(o), fmt(c), False, "osm"


def _address(tags: dict) -> tuple[str, str]:
    """(address string from OSM tags, ward). OSM address tags are often partial - this is best effort."""
    ward = tags.get("addr:subdistrict") or tags.get("addr:ward") or tags.get("addr:suburb") or ""
    parts = [
        " ".join(p for p in (tags.get("addr:housenumber"), tags.get("addr:street")) if p),
        ward,
        tags.get("addr:city") or tags.get("addr:province") or "",
    ]
    return ", ".join(p for p in parts if p), ward


def _is_alley(tags: dict) -> bool:
    """Heuristic: Vietnamese addresses like '12/3 Nguyen Van A' or 'hẻm/ngõ/kiệt' indicate an alley store."""
    hn = tags.get("addr:housenumber", "")
    street = tags.get("addr:street", "")
    return bool(re.search(r"\d+/\d+", hn) or re.search(r"\b(hẻm|hem|ngõ|ngách|kiệt|alley)\b", street, flags=re.I))


def _is_pharmacy(tags: dict) -> bool:
    return (tags.get("amenity") == "pharmacy" or tags.get("healthcare") == "pharmacy"
            or tags.get("shop") == "chemist") and "pharmacity" in " ".join(
        tags.get(k, "") for k in ("name", "brand", "operator")).lower()


def elements_to_dataframe(elements: list[dict]) -> pd.DataFrame:
    rows, seen = [], set()
    for el in elements:
        tags = el.get("tags") or {}
        if not _is_pharmacy(tags):
            continue
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lng = el.get("lon", (el.get("center") or {}).get("lon"))
        if lat is None or lng is None:
            continue
        eid = f"osm-{el.get('type', 'node')}-{el.get('id')}"
        name = tags.get("name") or tags.get("brand") or "Pharmacity"
        dup_key = (round(float(lat), 5), round(float(lng), 5), name.strip().lower())
        if eid in seen or dup_key in seen:
            continue
        seen.update({eid, dup_key})
        o, c, is24, src = parse_opening_hours(tags.get("opening_hours"))
        addr, ward = _address(tags)
        rows.append({
            "id": eid,
            "name": name,
            "lat": round(float(lat), 6),
            "lng": round(float(lng), 6),
            "open_time": o,
            "close_time": c,
            "is_24h": is24,
            "address_new": addr,
            "ward_new": ward,
            "area": ward or tags.get("addr:street", ""),
            "car_parking": False,       # not in OSM -> assumed False (survey to improve)
            "oneway_street": False,     # not in OSM -> assumed False (survey to improve)
            "access_type": "hem" if _is_alley(tags) else "mat_tien",
            "format": "24/7" if is24 else "Standard",
            "is_demo": False,
            "hours_source": src,
            "osm_url": f"https://www.openstreetmap.org/{el.get('type', 'node')}/{el.get('id')}",
        })
    cols = ["id", "name", "lat", "lng", "open_time", "close_time", "is_24h", "address_new", "ward_new", "area",
            "car_parking", "oneway_street", "access_type", "format", "is_demo", "hours_source", "osm_url"]
    return pd.DataFrame(rows, columns=cols).sort_values(["lat", "lng"]).reset_index(drop=True)


def refresh(out_path: str | Path, bbox=DEFAULT_BBOX, session=None, deadline: float = 120.0,
            elements: list[dict] | None = None) -> pd.DataFrame:
    """Download (or take `elements`) -> convert -> write CSV (UTF-8 with BOM so Excel shows Vietnamese). Returns the table."""
    els = elements if elements is not None else fetch_elements(bbox, session=session, deadline=deadline)
    df = elements_to_dataframe(els)
    if df.empty:
        raise RuntimeError("No Pharmacity stores found in this data / bounding box.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download real Pharmacity stores from OpenStreetMap.")
    ap.add_argument("--bbox", default=",".join(map(str, DEFAULT_BBOX)), help="south,west,north,east")
    ap.add_argument("--out", default="data/stores_hcmc_osm.csv")
    ap.add_argument("--from-json", help="convert a raw Overpass JSON export instead of downloading")
    ap.add_argument("--print-query", action="store_true", help="print the Overpass query (for overpass-turbo.eu) and exit")
    a = ap.parse_args(argv)
    bbox = tuple(float(x) for x in a.bbox.split(","))
    if len(bbox) != 4:
        print("--bbox needs 4 numbers: south,west,north,east", file=sys.stderr)
        return 2
    if a.print_query:
        print(build_query(bbox))
        return 0
    try:
        df = refresh(a.out, bbox, elements=elements_from_json_file(a.from_json) if a.from_json else None)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        print("Tip: use  python osm_stores.py --print-query  , run it on https://overpass-turbo.eu , export raw JSON, then "
              "python osm_stores.py --from-json export.json", file=sys.stderr)
        return 1
    n_osm = int((df["hours_source"] == "osm").sum())
    print(f"Saved {len(df)} stores to {a.out}")
    print(f"  24/7: {int(df['is_24h'].sum())} | opening hours from OSM: {n_osm} | assumed "
          f"{ASSUMED_HOURS[0]}-{ASSUMED_HOURS[1]}: {len(df) - n_osm}")
    print("  car_parking / oneway_street are NOT in OSM and default to False - survey them if they matter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
