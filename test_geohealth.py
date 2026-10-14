"""Offline tests for geohealth.py (no internet, Streamlit or Folium needed).

Run from the project root:  python tests/test_geohealth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import core
import geohealth as g

SC = g.Scenario()
THU_DUC = (10.8494, 106.7537)
NHA_BE = (10.7000, 106.7400)


# ---------------------------------------------------------------- levels
def test_aqi_levels_at_boundaries():
    assert g.aqi_level(50)["short"] == "Green"
    assert g.aqi_level(51)["short"] == "Yellow"
    assert g.aqi_level(100)["short"] == "Yellow"
    assert g.aqi_level(101)["short"] == "Orange"
    assert g.aqi_level(150)["short"] == "Orange"
    assert g.aqi_level(151)["short"] == "Red"
    assert g.aqi_level(301)["short"] == "Maroon"


def test_risk_levels_at_boundaries():
    assert [g.risk_level(v)["name"] for v in (0, 24.9, 25, 49.9, 50, 75, 100)] == \
        ["Low", "Low", "Moderate", "Moderate", "High", "Very high", "Very high"]


def test_colorize_matches_levels_and_shape():
    vals = np.array([[10.0, 75.0], [120.0, 400.0]])
    rgba = g.colorize("aqi", vals)
    assert rgba.shape == (2, 2, 4) and rgba.dtype == np.uint8
    assert tuple(rgba[1, 0, :3]) == g._hex_rgb("#FF7E00")      # 120 -> Orange
    assert tuple(rgba[1, 1, :3]) == g._hex_rgb("#7E0023")      # 400 -> Maroon
    risk = g.colorize("dengue", np.array([[10.0, 60.0]]))
    assert tuple(risk[0, 0, :3]) == g._hex_rgb("#2E8B6E") and tuple(risk[0, 1, :3]) == g._hex_rgb("#E8743B")


# ---------------------------------------------------------------- simulated fields
def test_hotspot_is_worse_than_clean_area():
    assert g.value_at("aqi", *THU_DUC, 14 * 60, SC) > g.value_at("aqi", *NHA_BE, 14 * 60, SC) + 40
    assert g.value_at("dengue", 10.7654, 106.6038, 0, SC) > g.value_at("dengue", *NHA_BE, 0, SC) + 40


def test_rush_hour_worse_than_night_and_scale_applies():
    assert g.value_at("aqi", *THU_DUC, 18 * 60, SC) > g.value_at("aqi", *THU_DUC, 2 * 60, SC)
    hazy = g.Scenario(aqi_scale=1.9)
    assert g.value_at("aqi", *THU_DUC, 14 * 60, hazy) > 1.8 * g.value_at("aqi", *THU_DUC, 14 * 60, SC) - 1


def test_field_grid_shape_and_orientation():
    la, ln = g.make_grid(ny=20, nx=30)
    assert la.shape == (20, 30) and la[0, 0] > la[-1, 0]       # row 0 = north
    v = g.field("aqi", la, ln, 8 * 60, SC)
    assert v.shape == (20, 30) and v.min() >= 0 and v.max() <= 500


# ---------------------------------------------------------------- live AQI (mocked)
class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


class _Sess:
    def __init__(self, payload): self.payload, self.params = payload, None
    def get(self, url, params=None, timeout=None):
        self.params = params
        return _Resp(self.payload)


def test_fetch_live_aqi_parses_list_and_skips_nulls():
    payload = [{"current": {"us_aqi": 80 + i}} for i in range(24)] + [{"current": {"us_aqi": None}}]
    live = g.fetch_live_aqi(session=_Sess(payload))
    assert len(live.vals) == 24 and live.pts.shape == (24, 2)


def test_fetch_live_aqi_too_few_values_raises():
    try:
        g.fetch_live_aqi(session=_Sess([{"current": {"us_aqi": None}}] * 25))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_live_overrides_simulation_with_idw():
    pts = np.array([[10.7, 106.6], [10.9, 106.8], [10.7, 106.8], [10.9, 106.6]])
    live = g.LiveAQI(pts, np.array([200.0, 200.0, 200.0, 200.0]), "12:00")
    sc = g.Scenario(live=live)
    assert abs(g.value_at("aqi", 10.8, 106.7, 0, sc) - 200.0) < 1e-6
    # uniform live data -> same value anywhere, and the simulated time-of-day profile no longer applies
    assert abs(g.value_at("aqi", *NHA_BE, 0, sc) - g.value_at("aqi", *THU_DUC, 8 * 60, sc)) < 1e-6


# ---------------------------------------------------------------- hazards and basket
def test_assess_thu_duc_afternoon_is_orange_or_worse():
    hz = {h.key: h for h in g.assess(*THU_DUC, 14 * 60, SC)}
    assert hz["aqi"].active and hz["aqi"].level_idx >= g.AQI_ACTIVE_IDX and hz["aqi"].advice


def test_thu_duc_scenario_triggers_two_hazards_and_merged_basket():
    hz = g.assess(*THU_DUC, 20 * 60, g.Scenario(aqi_scale=1.9))
    assert {h.key for h in hz if h.active} == {"aqi", "dengue"}
    keys = [b["key"] for b in g.build_basket(hz)]
    assert "n95" in keys and "repellent" in keys


def test_assess_clean_area_has_no_active_hazard():
    hz = g.assess(*NHA_BE, 14 * 60, g.Scenario(aqi_scale=0.7))
    assert not any(h.active for h in hz) and g.build_basket(hz) == []


def test_basket_merges_dedupes_and_marks_essentials():
    mk = lambda key, p: g.Hazard(key, key, 50, "x", "x", "#000", 3, True, p, "")
    basket = g.build_basket([mk("aqi", 0.5), mk("conj", 0.9)])
    keys = [b["key"] for b in basket]
    assert keys[0] == "saline_eye"                 # conj is more severe -> its items first
    assert len(keys) == len(set(keys))             # saline_eye appears once even though both baskets list it
    assert {b["key"]: b["essential"] for b in basket}["n95"] is True
    assert {b["key"]: b["essential"] for b in basket}["saline_nasal"] is False


# ---------------------------------------------------------------- stock
def test_stock_is_deterministic_and_pressure_lowers_availability():
    row = {"id": "s1"}
    assert g.stock_status(row, "n95", 0.3) == g.stock_status(row, "n95", 0.3)
    n = 400
    lo = sum(g.stock_status({"id": f"s{i}"}, "n95", 0.0)[0] for i in range(n)) / n
    hi = sum(g.stock_status({"id": f"s{i}"}, "n95", 1.0)[0] for i in range(n)) / n
    assert lo > hi + 0.2


def test_real_stock_column_overrides_simulation():
    assert g.stock_status({"id": "s1", "stock_n95": "yes"}, "n95", 1.0) == (True, False)
    assert g.stock_status({"id": "s1", "stock_n95": "0"}, "n95", 0.0) == (False, False)


# ---------------------------------------------------------------- store ranking
def _stores():
    base = dict(open_min=420.0, close_min=1320.0, is_24h=False, car_parking=True, oneway_street=False,
                access_type="mat_tien", area="", address_new="", is_demo=False)
    rows = [
        dict(id="near_out", name="Near (no stock)", lat=10.8500, lng=106.7540, stock_n95="0", stock_saline_eye="0", **base),
        dict(id="mid_ok", name="Mid (full stock)", lat=10.8540, lng=106.7560, stock_n95="1", stock_saline_eye="1",
             stock_saline_nasal="1", **base),
        dict(id="far_ok", name="Far", lat=10.9000, lng=106.8000, stock_n95="1", stock_saline_eye="1", **base),
        dict(id="closed", name="Closed at night", lat=10.8496, lng=106.7538, stock_n95="1", stock_saline_eye="1", **base),
    ]
    rows[3]["open_min"], rows[3]["close_min"] = 480.0, 1080.0   # closes 18:00
    return pd.DataFrame(rows)


def test_recommend_prefers_stocked_store_over_nearest_and_skips_closed():
    sc = g.Scenario(aqi_scale=1.9, dengue_scale=0.2)   # isolate the AQI hazard
    hz = g.assess(*THU_DUC, 20 * 60, sc)
    basket = g.build_basket(hz)
    assert any(b["key"] == "n95" for b in basket)
    opts, name, fell = g.recommend_stores(_stores(), THU_DUC, 20 * 60, "motorbike", basket)
    ids = [o.store["id"] for o in opts]
    assert "closed" not in ids and fell is False and name.startswith("Internal")
    assert ids[0] == "mid_ok"                                   # nearest store is out of N95 + saline
    near = next(o for o in opts if o.store["id"] == "near_out")
    assert near.missing_essential == 2 and near.cost > opts[0].cost


def test_headline_mentions_zone_store_distance_and_items():
    sc = g.Scenario(aqi_scale=1.9, dengue_scale=0.2)   # isolate the AQI hazard
    hz = g.assess(*THU_DUC, 20 * 60, sc)
    opts, _, _ = g.recommend_stores(_stores(), THU_DUC, 20 * 60, "motorbike", g.build_basket(hz))
    txt = g.headline("Thu Duc", hz, opts)
    assert "Thu Duc" in txt and "Mid (full stock)" in txt and "N95" in txt and "in stock" in txt
    assert "fine-dust" in txt


def test_headline_without_hazard_or_stores():
    hz = g.assess(*NHA_BE, 14 * 60, g.Scenario(aqi_scale=0.7))
    opts, _, _ = g.recommend_stores(_stores(), NHA_BE, 14 * 60, "motorbike", [])
    assert "No elevated" in g.headline("Nha Be", hz, opts)
    assert "No store is open" in g.headline("Nha Be", hz, [])


def test_nearest_area_and_helpers():
    assert g.nearest_area(*THU_DUC) == "Thu Duc"
    assert g.fmt_dist(0.5) == "500 m" and g.fmt_dist(2.34) == "2.3 km"
    assert abs(g.walk_minutes(0.5) - 6.0) < 1e-9
    assert g.area_ranking("aqi", 14 * 60, SC)[0]["Value"] >= g.area_ranking("aqi", 14 * 60, SC)[-1]["Value"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
