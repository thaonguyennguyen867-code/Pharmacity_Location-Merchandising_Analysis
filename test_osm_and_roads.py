"""Offline tests for the new parts: OSM import, real-road distances, circuity calibration.

Run from the project root:  python tests/test_osm_and_roads.py
(No internet, Streamlit or Folium needed - the network is mocked.)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import core
import osm_stores as osm


# ---------------------------------------------------------------- opening-hours parsing
def test_hours_24_7():
    assert osm.parse_opening_hours("24/7") == ("00:00", "24:00", True, "osm")
    assert osm.parse_opening_hours("Mo-Su 00:00-24:00")[2] is True


def test_hours_simple_and_overnight():
    assert osm.parse_opening_hours("Mo-Su 07:00-22:00") == ("07:00", "22:00", False, "osm")
    assert osm.parse_opening_hours("06:30-01:00") == ("06:30", "01:00", False, "osm")


def test_hours_split_shift_uses_union():
    assert osm.parse_opening_hours("Mo-Fr 08:00-12:00,14:00-20:00")[:2] == ("08:00", "20:00")


def test_hours_missing_or_unreadable_is_assumed():
    for v in (None, "", "by appointment", "sunrise-sunset"):
        o, c, is24, src = osm.parse_opening_hours(v)
        assert (o, c) == osm.ASSUMED_HOURS and not is24 and src == "assumed"


# ---------------------------------------------------------------- element conversion
ELEMENTS = [
    {"type": "node", "id": 1, "lat": 10.7769, "lon": 106.7009,
     "tags": {"amenity": "pharmacy", "name": "Pharmacity", "opening_hours": "24/7",
              "addr:housenumber": "12", "addr:street": "Nguyen Hue", "addr:subdistrict": "Ben Nghe"}},
    {"type": "way", "id": 2, "center": {"lat": 10.80, "lon": 106.65},
     "tags": {"amenity": "pharmacy", "brand": "Pharmacity", "addr:housenumber": "45/3", "addr:street": "Cach Mang Thang 8"}},
    {"type": "node", "id": 3, "lat": 10.80, "lon": 106.65,   # duplicate position+name of way 2 -> dropped
     "tags": {"amenity": "pharmacy", "brand": "Pharmacity", "name": "Pharmacity"}},
    {"type": "node", "id": 4, "lat": 10.79, "lon": 106.70,   # not a pharmacy (head office) -> dropped
     "tags": {"office": "company", "name": "Pharmacity Head Office"}},
    {"type": "node", "id": 5, "tags": {"amenity": "pharmacy", "name": "Pharmacity"}},  # no coordinates -> dropped
]


def test_elements_to_dataframe():
    df = osm.elements_to_dataframe(ELEMENTS)
    assert len(df) == 2
    a = df[df["id"] == "osm-node-1"].iloc[0]
    b = df[df["id"] == "osm-way-2"].iloc[0]
    assert a["is_24h"] and a["hours_source"] == "osm" and a["ward_new"] == "Ben Nghe"
    assert a["access_type"] == "mat_tien"
    assert b["access_type"] == "hem"            # '45/3' house number -> alley heuristic
    assert b["hours_source"] == "assumed" and not b["is_24h"]
    assert (b["lat"], b["lng"]) == (10.8, 106.65)


def test_csv_roundtrips_through_core_loader():
    df = osm.elements_to_dataframe(ELEMENTS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    stores, warns = core.load_stores(buf)
    assert warns == [] and len(stores) == 2
    assert stores.set_index("id").loc["osm-node-1", "is_24h"]
    assert core.open_mask(stores, 2 * 60).sum() == 1   # only the 24/7 store is open at 02:00


class _FakeResp:
    def __init__(self, payload, fail=False):
        self._p, self._f = payload, fail

    def raise_for_status(self):
        if self._f:
            raise RuntimeError("boom")

    def json(self):
        return self._p


class _FakeSession:
    def __init__(self, plan):
        self.plan, self.calls = list(plan), []

    def post(self, url, **kw):
        self.calls.append(url)
        return self.plan.pop(0)


def test_fetch_falls_back_to_second_mirror():
    sess = _FakeSession([_FakeResp({}, fail=True), _FakeResp({"elements": ELEMENTS})])
    els = osm.fetch_elements(urls=["https://a.test/x", "https://b.test/x"], session=sess)
    assert len(els) == len(ELEMENTS) and len(sess.calls) == 2


def test_fetch_raises_when_all_mirrors_fail():
    sess = _FakeSession([_FakeResp({}, fail=True), _FakeResp({}, fail=True)])
    try:
        osm.fetch_elements(urls=["https://a.test/x", "https://b.test/x"], session=sess)
    except RuntimeError as e:
        assert "unreachable" in str(e) and "a.test" in str(e) and "b.test" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_query_uses_indexed_tags_and_bbox():
    q = osm.build_query((10.6, 106.5, 10.9, 106.9))
    assert q.count("pharmacity") == 9 and "(10.6,106.5,10.9,106.9)" in q
    assert all(line.strip().startswith("nwr[") for line in q.splitlines() if "pharmacity" in line)


def test_deadline_stops_trying_more_mirrors():
    class Slow(_FakeSession):
        def post(self, url, **kw):
            self.calls.append(url)
            raise TimeoutError("slow")
    sess = Slow([])
    try:
        osm.fetch_elements(urls=[f"https://m{i}.test/x" for i in range(5)], session=sess, deadline=-1)
    except RuntimeError as e:
        assert "time budget" in str(e)
    assert len(sess.calls) == 1            # first mirror is always tried, then the budget stops the rest


def test_from_json_roundtrip(tmp_path=None):
    import json, tempfile
    d = Path(tempfile.mkdtemp())
    (d / "export.json").write_text(json.dumps({"elements": ELEMENTS}), encoding="utf-8")
    out = d / "stores.csv"
    assert osm.main(["--from-json", str(d / "export.json"), "--out", str(out)]) == 0
    stores, warns = core.load_stores(out)
    assert len(stores) == 2 and warns == []
    (d / "geo.json").write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    try:
        osm.elements_from_json_file(d / "geo.json")
    except ValueError as e:
        assert "raw OSM data" in str(e)
    else:
        raise AssertionError("expected ValueError for GeoJSON")


# ---------------------------------------------------------------- real-road distances (OSRM mocked)
def _stores(n=14, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": [f"s{i}" for i in range(n)], "name": [f"S{i}" for i in range(n)],
        "lat": 10.72 + rng.random(n) * 0.12, "lng": 106.62 + rng.random(n) * 0.14,
        "open_min": 420.0, "close_min": 1320.0, "is_24h": False, "car_parking": True, "oneway_street": False,
        "access_type": "mat_tien", "area": "", "address_new": "", "is_demo": False,
    })


def _fake_table(factor):
    """OSRM stand-in: road distance = haversine x factor, speed 20 km/h."""
    def table(base_url, pts, timeout):
        n = len(pts)
        dist = [[core.haversine_km(pts[i], pts[j]) * factor * 1000 for j in range(n)] for i in range(n)]
        dur = [[d / 1000 / 20 * 3600 for d in row] for row in dist]
        return tuple(map(tuple, dur)), tuple(map(tuple, dist))
    return table


def test_calibrate_circuity_recovers_factor():
    orig = core._osrm_table
    core._osrm_table = _fake_table(1.62)
    try:
        res = core.calibrate_circuity(_stores())
    finally:
        core._osrm_table = orig
    assert res is not None and abs(res[0] - 1.62) < 0.01 and res[1] >= 5


def test_calibrate_circuity_returns_none_offline():
    def boom(*a, **k):
        raise ConnectionError("offline")
    orig = core._osrm_table
    core._osrm_table = boom
    try:
        assert core.calibrate_circuity(_stores()) is None
    finally:
        core._osrm_table = orig


def test_osrm_router_reports_real_km_and_plan_has_eta_km():
    orig = core._osrm_table
    core._osrm_table = _fake_table(1.8)
    core.OSRMRouter._down_until = 0.0
    try:
        st = _stores()
        o, d = (10.75, 106.65), (10.82, 106.74)
        q = core.Query(origin=o, dest=d, mode="motorbike", now_min=14 * 60, urgency="medium")
        rec = core.recommend(st, q, router=core.OSRMRouter("http://fake"), top_k=3)
    finally:
        core._osrm_table = orig
    assert rec.router_name.startswith("OSRM") and not rec.fell_back
    best = rec.plans[0]
    # the first leg is origin -> store; its km must be the (mock) road km, i.e. 1.8x straight line
    s = (best.store["lat"], best.store["lng"])
    expected = core.haversine_km(o, s) * 1.8
    pos = best.insert_pos
    if pos == 0:
        assert abs(best.eta_km - expected) < 0.01


def test_offline_km_uses_current_circuity():
    r = core.HeuristicRouter()
    a, b = (10.77, 106.70), (10.80, 106.72)
    core.set_circuity(1.35)
    k1 = r.km(a, b)
    core.set_circuity(2.0)
    k2 = r.km(a, b)
    core.set_circuity(1.35)
    assert abs(k2 / k1 - 2.0 / 1.35) < 1e-9
    core.set_circuity(99)
    assert core.CIRCUITY == 2.5
    core.set_circuity(1.35)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
