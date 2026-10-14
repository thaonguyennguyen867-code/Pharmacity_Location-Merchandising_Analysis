"""
Pharmacity HCMC - Trip-Chaining Optimizer x T2M Urgency Matrix

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import core
import geohealth
import osm_stores
import viz

ROOT = Path(__file__).parent
DEMO_CSV = ROOT / "data" / "stores_hcmc_demo.csv"
OSM_CSV = ROOT / "data" / "stores_hcmc_osm.csv"
TIME_OPTIONS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
CUSTOM = "Type another address or coordinates…"
SRC_OSM, SRC_DEMO, SRC_UPLOAD = "Real stores (OpenStreetMap)", "Demo data (synthetic)", "Upload your CSV"
ROUTER_OSRM, ROUTER_LOCAL = "Real roads (OSRM, needs internet)", "Offline estimate (straight line x factor)"
DEFAULT_OSRM = "https://router.project-osrm.org"
DEFAULT_CIRCUITY = 1.35

st.set_page_config(page_title="Pharmacity HCMC - Trip-Chaining x T2M", page_icon="💊", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');
h1, h2, h3, h4, .stMarkdown p, .stMarkdown li, label, [data-testid="stMetricValue"],
[data-testid="stMetricLabel"], .t2m-banner, .t2m-card { font-family: 'Be Vietnam Pro', sans-serif; }
.t2m-banner { border-radius: 14px; padding: 20px 28px; display: flex; align-items: center; gap: 28px; margin: 4px 0 18px 0; }
.t2m-clock { font-size: 64px; font-weight: 300; line-height: 1; letter-spacing: -2px; font-variant-numeric: tabular-nums; }
.t2m-band { font-size: 18px; font-weight: 600; }
.t2m-sub { font-size: 14px; opacity: .85; margin-top: 4px; }
.t2m-card { border: 1px solid rgba(128,128,128,.30); border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; }
.t2m-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
             background: #F2B134; color: #3a2a00; margin-left: 10px; vertical-align: 2px; }
.t2m-chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.t2m-swatch { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin: 0 6px 0 16px; vertical-align: -1px; }
</style>
"""


# ------------------------------------------------------------------ UI helpers
def sky(minute: float) -> tuple[str, str, str]:
    """(time-of-day name, background gradient, text colour) - the banner changes with the simulated hour."""
    h = (minute % 1440) / 60
    if h >= 22 or h < 5:
        return "Late night", "linear-gradient(120deg,#0B1626,#1B3558)", "#EAF2FB"
    if h < 8:
        return "Early morning", "linear-gradient(120deg,#F2C6A0,#8FBBD9)", "#1B2A38"
    if h < 16:
        return "Daytime", "linear-gradient(120deg,#DCEFF0,#B9DCE2)", "#12313A"
    if h < 19.5:
        return "Late afternoon", "linear-gradient(120deg,#F4B26B,#B76E8F)", "#2A1A24"
    return "Evening", "linear-gradient(120deg,#2B3F66,#101C33)", "#EAF2FB"


def banner(now_min: float, stores: pd.DataFrame) -> None:
    band, grad, fg = sky(now_min)
    n_open = int(core.open_mask(stores, now_min).sum())
    n24 = int(stores["is_24h"].sum())
    st.markdown(
        f'<div class="t2m-banner" style="background:{grad};color:{fg}">'
        f'<div class="t2m-clock">{core.fmt_hhmm(now_min)}</div>'
        f'<div><div class="t2m-band">{band}</div>'
        f'<div class="t2m-sub">{n_open}/{len(stores)} stores open, {n24} of them 24/7</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def legend(items: list[tuple[str, str]]) -> None:
    html = "".join(f'<span class="t2m-swatch" style="background:{c}"></span>{t}' for c, t in items)
    st.markdown(f"<div style='font-size:13px;opacity:.9'>{html}</div>", unsafe_allow_html=True)


def show_map(m, height: int = 540, returned=None, key=None):
    """Render a Folium map. `returned` lists the objects to send back (e.g. ["last_clicked"]); returns that dict."""
    kw = dict(height=height, returned_objects=returned or [], key=key)
    try:
        return st_folium(m, use_container_width=True, **kw)
    except TypeError:  # newer streamlit-folium versions dropped use_container_width
        return st_folium(m, width=1100, **kw)


def place_picker(label: str, key: str, default: str):
    names = list(core.PLACES) + [CUSTOM]
    choice = st.selectbox(label, names, index=names.index(default), key=f"{key}_sel")
    if choice != CUSTOM:
        return choice, core.PLACES[choice]
    txt = st.text_input("Coordinates 'lat, lng' or an address", key=f"{key}_txt", placeholder="10.7769, 106.7009")
    if not txt.strip():
        st.info("Enter coordinates or an address to continue.")
        return None
    pt = core.parse_latlng(txt) or core.geocode(txt)
    if pt is None:
        st.error("Location not found. Enter coordinates like 10.7769, 106.7009 (address lookup needs internet).")
        return None
    return txt, pt


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float):
    """`mtime` is part of the cache key so a refreshed file is re-read."""
    return core.load_stores(path)


@st.cache_data(show_spinner=False)
def cached_grid(df: pd.DataFrame, step_km: float, circuity: float):
    """`circuity` is part of the cache key because grid distances depend on it."""
    return core.build_grid(df, step_km=step_km)


def apply_247_override(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    if not ids:
        return df
    df = df.copy()
    m = df["id"].isin(ids)
    df.loc[m, "is_24h"] = True
    df.loc[m, "open_min"] = 0.0
    df.loc[m, "close_min"] = 1440.0
    df.loc[m, "open_time"] = "00:00"
    df.loc[m, "close_time"] = "24:00"
    df.loc[m, "format"] = "24/7"
    return df


# ------------------------------------------------------------------ Tab 1: find a store
def render_find(stores, now_min, mode, urgency, store_dwell, via_dwell, osrm_url, tol_km):
    a, b = st.columns(2)
    with a:
        o = place_picker("Start (office)", "origin", "Bitexco Financial Tower")
    with b:
        d = place_picker("End (home)", "dest", "Bay Hien Intersection")
    if o is None or d is None:
        return

    via_names: list[str] = []
    if urgency == "high":
        st.caption("High urgency: go straight from where you are to the store, ignoring mid-trip stops.")
        st.info("If there are danger signs such as difficulty breathing, chest pain, high fever with convulsions or "
                "confusion, call 115 or go to the nearest emergency facility instead of a pharmacy.")
    else:
        via_names = st.multiselect(
            "Mid-trip stops (supermarket, school…) - optional, visited in the order you pick",
            list(core.PLACES), max_selections=2, key="via",
        )
    require_parking = False
    if mode == "car":
        require_parking = st.checkbox("Only stores with car parking (if unchecked: stores without parking get extra minutes)", key="parking")

    q = core.Query(
        origin=o[1], dest=d[1], mode=mode, now_min=float(now_min), urgency=urgency,
        via=[core.PLACES[n] for n in via_names], via_dwell=float(via_dwell),
        store_dwell=float(store_dwell), require_parking=require_parking,
    )
    router = core.OSRMRouter(osrm_url) if osrm_url else core.HeuristicRouter()
    rec = core.recommend(stores, q, router=router, top_k=5)

    if osrm_url and rec.fell_back:
        st.warning("Could not reach OSRM (you may be offline). Results use the offline estimate.")
    st.caption(f"Router: {rec.router_name}")

    if not rec.plans:
        st.warning("No store is still open when you arrive. Try another simulated time, vehicle, or drop the parking condition.")
        return

    best = rec.plans[0]
    s = best.store
    badge = '<span class="t2m-badge">24/7</span>' if s["is_24h"] else ""
    addr = ". ".join(x for x in (s["area"], s["address_new"]) if x)
    st.markdown(
        f'<div class="t2m-card"><div style="font-size:19px;font-weight:600">{s["name"]}{badge}</div>'
        f'<div style="opacity:.8;margin-top:2px">{addr}</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Arrive at", core.fmt_hhmm(best.arrival_min))
    c2.metric("Travel to store", f"{best.eta_min:.0f} min")
    c3.metric("Road distance to store", f"{best.eta_km:.1f} km")
    c4.metric("Detour vs. original route", f"+{best.detour_min:.0f} min")
    c5.metric("Store closes at", "24/7" if best.mins_to_close is None else core.fmt_hhmm(best.arrival_min + best.mins_to_close))
    for n in best.notes:
        st.caption(f"Note: {n}")

    poly = router.polyline(best.waypoints, mode)
    base_poly = router.polyline(rec.chain, mode)
    open_now = core.open_mask(stores, now_min)
    hot = st.toggle("Show congestion hotspots (assumed)", value=False, key="hot")
    show_map(viz.result_map(stores, open_now, q, rec, poly, base_poly, tol_km, show_hotspots=hot))
    legend([(viz.LAMP, "24/7"), (viz.OPEN, "open"), (viz.CLOSED, "closed"), (viz.ROUTE, "route with stop"),
            ("#9AA5AB", f"dashed circle: acceptable radius {tol_km:.1f} km")])
    if not router.online or rec.fell_back:
        st.caption("Lines are straight between points; choose 'Real roads (OSRM)' in the sidebar to draw the real road route.")

    rows = [{
        "Rank": i, "Store": p.store["name"].replace("[DEMO] ", ""), "Area": p.store["area"],
        "Hours": core.hours_label(p.store), "Arrive": core.fmt_hhmm(p.arrival_min),
        "ETA (min)": round(p.eta_min, 1), "Road km": round(p.eta_km, 2), "Detour (min)": round(p.detour_min, 1),
        "Access penalty (min)": round(p.penalty_min, 1), "Position on A-B axis (%)": round(p.progress * 100),
        "Off-axis (km)": round(p.lateral_km, 2), "Cost score": round(p.cost, 1),
    } for i, p in enumerate(rec.plans, start=1)]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if rec.excluded:
        with st.expander(f"Excluded stores ({len(rec.excluded)})"):
            reasons = pd.Series([r for _, r in rec.excluded]).value_counts().rename_axis("Reason").reset_index(name="Stores")
            st.dataframe(reasons, hide_index=True)
    with st.expander("How the cost score works"):
        lam = core.URGENCY[urgency]["lam"]
        st.markdown(
            f"**Cost = {lam:.2f} x ETA + {1 - lam:.2f} x detour + access penalty** (minutes, lower is better).\n\n"
            "ETA is the travel time from departure to the store; detour is the extra travel minutes compared with driving "
            "the original route directly. Low urgency favours stores that are on the way, High urgency favours arriving fast. "
            "A store is excluded if it is closed (or has no time left to shop) when you arrive, not when you press search. "
            "All speed, rush-hour, hotspot and penalty coefficients are assumptions in `core.py`."
        )


# ------------------------------------------------------------------ Tab 2: night analysis
def render_night(stores, base_km, mults):
    st.subheader("How much does the network thin out after dark?")
    if stores["is_demo"].all():
        st.warning("You are using synthetic DEMO data. The numbers below are only for trying the app, not for the report.")
    if "hours_source" in stores.columns:
        n_assumed = int((stores["hours_source"] == "assumed").sum())
        if n_assumed:
            st.warning(f"{n_assumed}/{len(stores)} stores have no opening hours in OpenStreetMap, so "
                       f"{osm_stores.ASSUMED_HOURS[0]}-{osm_stores.ASSUMED_HOURS[1]} is assumed. Night results depend heavily on this: "
                       "mark the genuine 24/7 stores in the sidebar override, or edit the CSV with verified hours.")
    c1, c2 = st.columns(2)
    day_str = c1.selectbox("Daytime hour to compare", TIME_OPTIONS, index=TIME_OPTIONS.index("14:00"), key="day_cmp")
    night_str = c2.selectbox("Night hour to compare", TIME_OPTIONS, index=TIME_OPTIONS.index("02:00"), key="night_cmp")
    day_min, night_min = core.to_minutes(day_str), core.to_minutes(night_str)

    step = 1.0 if len(stores) <= 100 else 1.5  # coarser grid for big datasets keeps maps responsive
    grid, dmat = cached_grid(stores, step, core.CIRCUITY)
    sm = core.summary_stats(dmat, stores, base_km, day_min, night_min)
    if sm["open_night"] == 0:
        st.warning(f"No store is open at {night_str} in this dataset, so there is nothing to compare. If some stores are "
                   "really 24/7, mark them in the sidebar override (or set open 00:00 / close 24:00 in the CSV).")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stores open", f"{sm['open_night']}/{sm['n_total']}", f"{sm['open_night'] - sm['open_day']} vs {day_str}")
    k2.metric("Median distance to nearest open store", f"{sm['median_night_km']:.1f} km",
              f"+{sm['median_night_km'] - sm['median_day_km']:.1f} km vs {day_str}", delta_color="inverse")
    k3.metric(f"Coverage within {base_km:.1f} km", f"{sm['coverage_night']:.0f}%", f"{sm['coverage_night'] - sm['coverage_day']:.0f} pts vs {day_str}")
    req = sm["required_mult"]
    k4.metric("Travel multiple needed to keep daytime coverage", "above x10" if req is None else f"x{req:g}")

    ratio_txt = "" if sm["median_ratio"] != sm["median_ratio"] else f" ({sm['median_ratio']:.1f}x)"
    need_txt = "more than x10" if req is None else f"x{req:g}"
    verdict = ""
    if req is not None:
        verdict = f" That is {'within' if req <= mults['high'] else 'beyond'} the x{mults['high']:g} hypothesis for high urgency."
    st.markdown(
        f"At {night_str}, {sm['open_night']}/{sm['n_total']} stores are open (24/7 stores are {sm['share_24h']:.0f}% of the network). "
        f"The median distance to the nearest open store rises from {sm['median_day_km']:.1f} to {sm['median_night_km']:.1f} km{ratio_txt}. "
        f"For night coverage to match daytime coverage at a {base_km:.1f} km radius, customers must be willing to travel {need_txt} their usual distance.{verdict}"
    )

    prof = core.hourly_profile(dmat, stores, base_km, {"x1": 1.0, "med": mults["medium"], "high": mults["high"]})
    cov = prof.set_index("hour")[["coverage_x1", "coverage_med", "coverage_high"]].rename(columns={
        "coverage_x1": f"Normal radius ({base_km:.1f} km)",
        "coverage_med": f"Medium urgency (x{mults['medium']:g})",
        "coverage_high": f"High urgency (x{mults['high']:g})",
    })
    st.markdown("**Coverage (% of area with an open store inside the acceptable radius) by hour**")
    st.line_chart(cov)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Stores open by hour**")
        st.line_chart(prof.set_index("hour")[["stores_open"]].rename(columns={"stores_open": "Stores open"}))
    with g2:
        st.markdown("**Distance to the nearest open store (km)**")
        st.line_chart(prof.set_index("hour")[["median_km", "p90_km"]].rename(columns={"median_km": "Median", "p90_km": "90th percentile"}))

    st.markdown("**24/7 stores 'inherit' the catchment of closed stores**")
    exp = core.catchment_expansion(dmat, stores, day_min, night_min, step_km=step)
    if exp.empty:
        st.info("The data contains no 24/7 stores.")
    else:
        total_ratio = exp["area_night_km2"].sum() / max(exp["area_day_km2"].sum(), 1e-9)
        st.metric("Catchment of the 24/7 group expands (night area / day area)", f"x{total_ratio:.1f}")
        st.dataframe(exp.rename(columns={
            "id": "ID", "name": "Store", "area": "Area", "area_day_km2": f"Nearest-store area at {day_str} (km²)",
            "area_night_km2": f"At {night_str} (km²)", "expansion": "Expansion (x)"}).round(2), hide_index=True)

    heat_str = st.select_slider("Map of distance to the nearest open store, at", options=TIME_OPTIONS, value=night_str, key="heat_time")
    hmin = core.to_minutes(heat_str)
    hmask = core.open_mask(stores, hmin)
    near = core.nearest_open_km(dmat, hmask)
    show_map(viz.heat_map(grid, near, stores, hmask, base_km, base_km * mults["high"], core.night_weight(hmin) >= 0.5), height=500)
    legend([(viz.OPEN, f"within {base_km:.1f} km"), (viz.LAMP, f"only within x{mults['high']:g} radius"),
            (viz.ALERT, "beyond even the emergency radius"), ("#FFFFFF", "open store (dark outline)")])

    with st.expander("How to read these numbers in the report"):
        st.markdown(
            "- **What the data can prove (supply side)**: when stores close, how much the distance to the nearest open store "
            "and the catchment of each 24/7 store change. This is computed directly from the store list and opening hours.\n"
            "- **What is an assumption (demand side)**: the x1.5 / x3 multipliers in the sidebar are customers' willingness to travel. "
            "The app cannot prove them; a customer survey or night-time transaction data is needed. The app gives the multiplier "
            "*required* to compensate, which you compare with real-world evidence.\n"
            f"- **Limits**: coverage is by area (not population), distance is straight line x a road factor (currently x{core.CIRCUITY:.2f}; "
            "calibrate it in the sidebar), and there is no night revenue or operating cost per store."
        )


# ------------------------------------------------------------------ Tab 3: Geo-Health map
GEO_AQI_SCEN = {"Clean day (x0.7)": 0.7, "Typical (x1.0)": 1.0, "Hazy (x1.4)": 1.4, "Severe haze (x1.9)": 1.9}
GEO_OUTBREAK = {"Low (x0.5)": 0.5, "Typical (x1.0)": 1.0, "Outbreak (x1.4)": 1.4}


@st.cache_data(ttl=900, show_spinner=False)
def cached_live_aqi():
    return geohealth.fetch_live_aqi()


def _chip_fg(hex_color: str) -> str:
    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return "#1a1a1a" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def render_geo(stores, now_min, mode, osrm_url, store_dwell):
    st.subheader("Geo-Health map: what does the local environment mean for health right now?")
    st.warning("Prototype data: the dengue and conjunctivitis layers, the simulated AQI and the store inventory are "
               "**synthetic scenarios**, not real surveillance or stock. Only the optional live AQI and the store "
               "locations are real. See 'How this works' below.")

    c1, c2 = st.columns([3, 2])
    layer = c1.radio("Map layer", list(geohealth.LAYERS), format_func=geohealth.LAYERS.get, horizontal=True, key="geo_layer")
    live_on = c2.toggle("Use live AQI (Open-Meteo, real, coarse ~10 km)", value=False, key="geo_live",
                        help="Replaces the simulated AQI with current model data. Needs internet; ignores the simulated time.")
    live = None
    if live_on:
        try:
            live = cached_live_aqi()
            st.caption(f"Live AQI loaded at {live.fetched_at} from {len(live.vals)} grid points (interpolated between them).")
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not load live AQI ({e}). Using the simulation instead.")
    k1, k2, k3 = st.columns(3)
    aqi_scale = GEO_AQI_SCEN[k1.select_slider("AQI scenario (simulation)", options=list(GEO_AQI_SCEN), value="Typical (x1.0)",
                                              key="geo_aqi_scn", disabled=live is not None)]
    dengue_scale = GEO_OUTBREAK[k2.select_slider("Dengue scenario", options=list(GEO_OUTBREAK), value="Typical (x1.0)", key="geo_dengue_scn")]
    conj_scale = GEO_OUTBREAK[k3.select_slider("Conjunctivitis scenario", options=list(GEO_OUTBREAK), value="Typical (x1.0)", key="geo_conj_scn")]
    sc = geohealth.Scenario(aqi_scale, dengue_scale, conj_scale, live)

    loc_mode = st.radio("Where are you?", ["Click on the map", "Pick a landmark / type an address"], horizontal=True, key="geo_locmode")
    if loc_mode.startswith("Click"):
        loc = st.session_state.get("geo_click") or core.PLACES["Thu Duc Intersection"]
    else:
        pick = place_picker("Your location", "geo_loc", "Thu Duc Intersection")
        if pick is None:
            return
        loc = pick[1]

    area = geohealth.nearest_area(*loc)
    hazards = geohealth.assess(loc[0], loc[1], now_min, sc)
    basket = geohealth.build_basket(hazards)
    router = core.OSRMRouter(osrm_url) if osrm_url else core.HeuristicRouter()
    options, router_name, fell_back = geohealth.recommend_stores(
        stores, loc, now_min, mode, basket, router=router, store_dwell=float(store_dwell))
    route_poly = router.polyline([loc, (options[0].store["lat"], options[0].store["lng"])], mode) if options else None

    la, ln = geohealth.make_grid()
    rgba = geohealth.colorize(layer, geohealth.field(layer, la, ln, now_min, sc))
    m = viz.geo_map(rgba, geohealth.GEO_BBOX, loc, stores, options, route_poly, core.night_weight(now_min) >= 0.5)
    res = show_map(m, height=560, returned=["last_clicked"], key="geo_map")
    legend(geohealth.legend_items(layer))
    if loc_mode.startswith("Click"):
        st.caption("Click anywhere on the heat map to move the YOU marker; the advice below updates.")
        click = res.get("last_clicked") if isinstance(res, dict) else None
        if click:
            new = (round(float(click["lat"]), 5), round(float(click["lng"]), 5))
            if new != st.session_state.get("geo_click"):
                st.session_state["geo_click"] = new
                st.rerun()

    # ---- hazard cards at your location
    st.markdown(f"**At your location: {area}** (nearest area centre, approximate) at {core.fmt_hhmm(now_min)}")
    for col, h in zip(st.columns(3), hazards):
        name = f"{h.short} - {h.level_name}" if h.key == "aqi" else h.level_name
        flag = "advice active" if h.active else "no action needed"
        col.markdown(
            f'<div class="t2m-card"><div style="font-size:13px;opacity:.8">{h.label}</div>'
            f'<div style="font-size:30px;font-weight:600">{h.value:.0f}</div>'
            f'<span class="t2m-chip" style="background:{h.color};color:{_chip_fg(h.color)}">{name}</span>'
            f'<div style="font-size:12px;opacity:.7;margin-top:4px">{flag}</div></div>',
            unsafe_allow_html=True,
        )

    # ---- recommendation
    headline = geohealth.headline(area, hazards, options)
    (st.success if options else st.warning)(headline)
    for h in hazards:
        if h.active and h.advice:
            st.markdown(f"- **{h.label}:** {h.advice}")
    st.caption(geohealth.DISCLAIMER)

    if options:
        if basket:
            top = options[:3]
            rows = []
            for b in basket:
                row = {"Item": b["name"], "Priority": "Essential" if b["essential"] else "Nice to have", "For": b["reason"]}
                for rank, o in enumerate(top, start=1):
                    ok = next(ok for k, _, ok in o.items if k == b["key"])
                    row[f"#{rank} has it"] = "yes" if ok else "no"
                rows.append(row)
            st.markdown("**Recommended basket and stock at the top stores**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        srows = []
        for rank, o in enumerate(options, start=1):
            have = sum(1 for _, _, ok in o.items if ok)
            srows.append({
                "Rank": rank, "Store": o.store["name"].replace("[DEMO] ", ""),
                "Area": o.store.get("area") or o.store.get("address_new", ""),
                "Distance": geohealth.fmt_dist(o.km), "Travel (min)": round(o.eta_min, 1),
                "Walk (min)": round(geohealth.walk_minutes(o.km)) if o.km <= 1.2 else "-",
                "Hours": core.hours_label(o.store),
                "Basket in stock": f"{have}/{len(o.items)}" if o.items else "-",
            })
        st.dataframe(pd.DataFrame(srows), hide_index=True)
        st.caption(f"Router: {router_name}" + (" (OSRM unreachable, offline estimate used)" if fell_back else ""))
        if basket and any(o.simulated_stock for o in options):
            keys = ", ".join(f"`stock_{k}`" for k in geohealth.PRODUCTS)
            st.caption(f"Stock is SIMULATED (more stock-outs where local demand is high). To use real inventory, add columns to "
                       f"your stores CSV, e.g. {keys} (yes/no).")

    with st.expander(f"Worst areas right now: {geohealth.LAYERS[layer]}"):
        st.dataframe(pd.DataFrame(geohealth.area_ranking(layer, now_min, sc)[:8]), hide_index=True)
    with st.expander("How this works (and what is simulated)"):
        st.markdown(
            "- **Heat layers**: AQI follows the US EPA bands (Orange = 101-150 = unhealthy for sensitive groups). The simulated AQI is a baseline "
            "plus illustrative hotspots (traffic, industry), scaled by a time-of-day profile and the scenario. Dengue and conjunctivitis are "
            "0-100 risk indices from illustrative hotspots. **None of these is surveillance data**; replace `AQI_SOURCES`, `DENGUE_SOURCES` "
            "and `CONJ_SOURCES` in `geohealth.py` with real feeds (e.g. HCDC bulletins) before drawing conclusions.\n"
            "- **Live AQI** (optional) comes from Open-Meteo's air-quality model on a coarse grid and is interpolated, so it cannot show "
            "street-level differences.\n"
            "- **Advice**: a hazard becomes *active* at AQI Orange or worse, or a Moderate+ risk index. Active hazards add products to the basket "
            "(masks and saline for AQI, repellent/paracetamol/ORS for dengue, saline and hand hygiene for pink eye).\n"
            "- **Store ranking**: only stores that are open when you would arrive; cost = travel minutes + 6 per missing essential item + 2 per "
            "missing other item, so a slightly farther store that has everything can beat the nearest one that does not.\n"
            "- **Retail angle**: where hazards and stock-outs coincide is where a store format with deeper health-context stock "
            "(or 24/7 hours) would matter most."
        )


# ------------------------------------------------------------------ Tab 4: data
def render_data(stores, template_bytes: bytes | None):
    st.subheader("Stores in use")
    st.caption(f"{len(stores)} stores, {int(stores['is_24h'].sum())} of them 24/7.")
    st.dataframe(stores.drop(columns=["open_min", "close_min"]), hide_index=True)
    cols = st.columns(2)
    cols[0].download_button("Download current stores (CSV)",
                            stores.drop(columns=["open_min", "close_min"]).to_csv(index=False).encode("utf-8-sig"),
                            file_name="stores_current.csv", mime="text/csv", key="dl_cur")
    if template_bytes:
        cols[1].download_button("Download sample CSV template", template_bytes, file_name="stores_template.csv",
                                mime="text/csv", key="dl_tpl")
    st.markdown(
        "**Required columns**: `id`, `name`, `lat`, `lng`, `open_time`, `close_time` (HH:MM; overnight such as 06:30-01:00 is supported; 24/7 = 00:00-24:00).\n\n"
        "**Optional columns**: `area`, `address_new` (new-style address: house no., street, ward/commune, Ho Chi Minh City), `ward_new`, `is_24h`, "
        "`car_parking`, `oneway_street`, `access_type` (`mat_tien` = street frontage or `hem` = alley), `format`, `is_demo`.\n\n"
        "**From OpenStreetMap**: coordinates and (where mapped) opening hours are real. `car_parking` and `oneway_street` are not in OSM and default to "
        "`False`; `access_type` is inferred from the address. `hours_source` tells you whether hours are `osm` or `assumed`."
    )


# ------------------------------------------------------------------ main
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Data and assumptions")
        source = st.radio("Store data source", [SRC_OSM, SRC_DEMO, SRC_UPLOAD], key="src")
        upload = st.file_uploader("Stores CSV", type="csv", key="upload") if source == SRC_UPLOAD else None
        if source == SRC_OSM:
            if st.button("Download / refresh from OpenStreetMap", key="osm_btn"):
                with st.spinner("Querying the Overpass API…"):
                    try:
                        fresh = osm_stores.refresh(OSM_CSV)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not download: {e}")
                    else:
                        st.cache_data.clear()
                        st.success(f"Saved {len(fresh)} real stores to data/{OSM_CSV.name}")
        router_choice = st.radio("Router", [ROUTER_OSRM, ROUTER_LOCAL], key="router")
        osrm_url = st.text_input("OSRM base URL", DEFAULT_OSRM, key="osrm_url") if router_choice == ROUTER_OSRM else None
        store_dwell = st.number_input("Shopping time at the store (min)", min_value=2, max_value=30, value=8, step=1, key="store_dwell")
        via_dwell = st.number_input("Time at each mid-trip stop (min)", min_value=0, max_value=90, value=15, step=5, key="via_dwell")
        st.markdown("**Demand assumptions (need validation)**")
        base_km = st.slider("Normal acceptable radius (road km)", min_value=0.5, max_value=4.0, value=1.5, step=0.1, key="base_km")
        m_med = st.number_input("Willingness-to-travel multiple: medium urgency", min_value=1.0, max_value=6.0,
                                value=float(core.URGENCY["medium"]["tol_mult"]), step=0.1, key="m_med")
        m_high = st.number_input("Willingness-to-travel multiple: high urgency", min_value=1.0, max_value=6.0,
                                 value=float(core.URGENCY["high"]["tol_mult"]), step=0.1, key="m_high")

    # ---- load stores
    stores = warns = None
    if source == SRC_OSM:
        if not OSM_CSV.exists() and not st.session_state.get("osm_autotried"):
            # first run with no data file: try to download the real stores automatically (once per session)
            st.session_state["osm_autotried"] = True
            with st.spinner("First run: downloading real Pharmacity stores from OpenStreetMap…"):
                try:
                    osm_stores.refresh(OSM_CSV)
                except Exception as e:  # noqa: BLE001
                    st.sidebar.error(f"Automatic download failed: {e}")
        if OSM_CSV.exists():
            stores, warns = load_csv(str(OSM_CSV), OSM_CSV.stat().st_mtime)
        else:
            st.sidebar.info("No real-store file yet. Click 'Download / refresh from OpenStreetMap' above (needs internet).")
    elif upload is not None:
        try:
            stores, warns = core.load_stores(io.BytesIO(upload.getvalue()))
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Could not read the CSV: {e}.")
    if stores is None and DEMO_CSV.exists():
        stores, warns = load_csv(str(DEMO_CSV), DEMO_CSV.stat().st_mtime)
    if stores is None:
        st.title("Pharmacity HCMC: which store, and when?")
        st.error(
            "No store data found yet. Do one of these:\n\n"
            "1. Click **Download / refresh from OpenStreetMap** in the sidebar (needs internet), or run `python osm_stores.py`.\n"
            f"2. Put your own file at `{DEMO_CSV}` or upload a CSV in the sidebar."
        )
        st.stop()
    for w in warns:
        st.sidebar.warning(w)

    with st.sidebar:
        labels = {f"{r.name} | {r.area or r.address_new or r.id} [{r.id}]": r.id for r in stores.itertuples()}
        picked = st.multiselect("Override: mark these stores as 24/7", list(labels), key="ovr_247",
                                help="Use when your data lacks opening hours but you know a store is open all night.")
        st.markdown("**Distance model (offline estimate and night analysis)**")
        if "circuity" not in st.session_state:
            st.session_state["circuity"] = DEFAULT_CIRCUITY
        if st.button("Calibrate road factor from real roads (OSRM)", key="calib"):
            res = core.calibrate_circuity(stores, osrm_url or DEFAULT_OSRM)
            if res is None:
                st.error("Could not calibrate (OSRM unreachable or too few store pairs).")
            else:
                st.session_state["circuity"] = res[0]
                st.success(f"Road factor = {res[0]:.2f} (median of {res[1]} store pairs)")
        core.set_circuity(st.session_state["circuity"])
        st.caption(f"Road distance ≈ straight line x {core.CIRCUITY:.2f}")

    stores = apply_247_override(stores, [labels[p] for p in picked if p in labels])

    st.title("Pharmacity HCMC: which store, and when?")
    st.caption("Three ideas in one app: the Trip-Chaining Optimizer (find the store that is on your way), "
               "the T2M Urgency Matrix (opening hours, time-of-day ETA and urgency level) "
               "and the Geo-Health Contextual Map (local air quality and outbreak risk -> what to buy, where).")

    c1, c2, c3 = st.columns([3, 2, 3])
    with c1:
        time_str = st.select_slider("Simulated time (when you set off)", options=TIME_OPTIONS, value="08:00", key="time")
    with c2:
        mode = {"Motorbike": "motorbike", "Car": "car"}[st.radio("Vehicle", ["Motorbike", "Car"], horizontal=True, key="mode")]
    with c3:
        urgency = st.radio("Urgency", ["low", "medium", "high"], format_func=lambda k: core.URGENCY[k]["label"],
                           horizontal=True, key="urgency")
        st.caption(core.URGENCY[urgency]["examples"])

    now_min = core.to_minutes(time_str)
    mults = {"low": 1.0, "medium": float(m_med), "high": float(m_high)}
    banner(now_min, stores)

    tab_find, tab_night, tab_geo, tab_data = st.tabs(["Find a store", "Night analysis", "Geo-Health map", "Data"])
    with tab_find:
        render_find(stores, now_min, mode, urgency, store_dwell, via_dwell, osrm_url, base_km * mults[urgency])
    with tab_night:
        render_night(stores, base_km, mults)
    with tab_geo:
        render_geo(stores, now_min, mode, osrm_url, store_dwell)
    with tab_data:
        render_data(stores, DEMO_CSV.read_bytes() if DEMO_CSV.exists() else None)


main()
