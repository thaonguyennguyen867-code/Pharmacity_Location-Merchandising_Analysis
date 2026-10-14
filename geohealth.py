"""
geohealth.py - "Geo-Health Contextual Map": local environmental health context -> product + store advice.

Pure logic (no Streamlit/Folium). Three environmental layers over HCMC:
  - AQI (air quality)            : SIMULATED by default; optional LIVE data from Open-Meteo (real, coarse ~10 km model)
  - Dengue fever risk (0-100)    : SIMULATED scenario (no open real-time source is wired in)
  - Conjunctivitis risk (0-100)  : SIMULATED scenario

Given the position of the user on the map it assesses each hazard, builds a recommended product basket,
and ranks nearby OPEN stores by travel time + how many basket items are in stock.

Everything marked SIMULATED is an illustrative scenario for the prototype, NOT surveillance data:
hotspots, amplitudes, the time-of-day AQI profile and the store inventory. Replace them with real feeds
(HCDC dengue bulletins, an AQI API, Pharmacity inventory) before drawing conclusions.
This is general consumer information, not medical advice.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import requests

import core

LatLng = tuple[float, float]

GEO_BBOX = (10.62, 106.52, 10.95, 106.90)  # south, west, north, east (HCMC urban core)
LAYERS = {
    "aqi": "Air quality (AQI)",
    "dengue": "Dengue fever risk",
    "conj": "Conjunctivitis (pink eye) risk",
}

# ====================================================================================
# 1. Levels
# ====================================================================================
AQI_LEVELS = [  # US EPA AQI categories
    dict(max=50, name="Good", short="Green", color="#00E400"),
    dict(max=100, name="Moderate", short="Yellow", color="#FFFF00"),
    dict(max=150, name="Unhealthy for sensitive groups", short="Orange", color="#FF7E00"),
    dict(max=200, name="Unhealthy", short="Red", color="#FF0000"),
    dict(max=300, name="Very unhealthy", short="Purple", color="#8F3F97"),
    dict(max=float("inf"), name="Hazardous", short="Maroon", color="#7E0023"),
]
RISK_LEVELS = [  # 0-100 risk index
    dict(min=0, name="Low", short="Low", color="#2E8B6E"),
    dict(min=25, name="Moderate", short="Moderate", color="#F2B134"),
    dict(min=50, name="High", short="High", color="#E8743B"),
    dict(min=75, name="Very high", short="Very high", color="#D1495B"),
]
AQI_ACTIVE_IDX = 2    # Orange or worse triggers product advice
RISK_ACTIVE_IDX = 1   # Moderate or worse triggers product advice


def aqi_level(v: float) -> dict:
    idx = next(i for i, lv in enumerate(AQI_LEVELS) if v <= lv["max"])
    return {**AQI_LEVELS[idx], "idx": idx}


def risk_level(v: float) -> dict:
    idx = max(i for i, lv in enumerate(RISK_LEVELS) if v >= lv["min"])
    return {**RISK_LEVELS[idx], "idx": idx}


def level_of(layer: str, v: float) -> dict:
    return aqi_level(v) if layer == "aqi" else risk_level(v)


def legend_items(layer: str) -> list[tuple[str, str]]:
    if layer == "aqi":
        out, lo = [], 0
        for lv in AQI_LEVELS:
            if lv["max"] == float("inf"):
                out.append((lv["color"], f"{lv['short']} {lo}+"))
            else:
                out.append((lv["color"], f"{lv['short']} {lo}-{int(lv['max'])}"))
                lo = int(lv["max"]) + 1
        return out
    return [(lv["color"], f"{lv['name']} (>= {lv['min']})") for lv in RISK_LEVELS]


# ====================================================================================
# 2. Simulated environment (ILLUSTRATIVE scenario, not data)
# ====================================================================================
# name, lat, lng, sigma km, amplitude
AQI_SOURCES = [
    ("Thu Duc industrial/traffic", 10.8494, 106.7537, 4.0, 70),
    ("Tan Binh airport / Cong Hoa", 10.8150, 106.6650, 3.0, 45),
    ("Binh Tan industrial", 10.7654, 106.6038, 4.0, 60),
    ("District 12", 10.8672, 106.6414, 3.5, 40),
    ("Hoc Mon", 10.8830, 106.5920, 4.0, 35),
    ("Binh Chanh", 10.6874, 106.5933, 4.0, 30),
    ("Cholon dense core", 10.7500, 106.6500, 2.5, 35),
    ("Hang Xanh junction", 10.8015, 106.7110, 2.5, 35),
    ("Riverside / Nha Be (cleaner)", 10.7000, 106.7400, 3.0, -15),
]
AQI_BASE = 45.0
# AQI multiplier by hour (traffic peaks + early-morning inversion)
_AQI_TOD = [
    0.85, 0.82, 0.80, 0.80, 0.85, 0.95,  # 00-05
    1.10, 1.25, 1.25, 1.10, 1.00, 0.95,  # 06-11
    0.95, 0.92, 0.92, 0.95, 1.05, 1.20,  # 12-17
    1.30, 1.25, 1.10, 1.00, 0.92, 0.88,  # 18-23
]
DENGUE_SOURCES = [
    ("Binh Tan", 10.7654, 106.6038, 3.0, 80),
    ("Thu Duc", 10.8494, 106.7537, 3.5, 70),
    ("District 12", 10.8672, 106.6414, 3.0, 65),
    ("Go Vap", 10.8386, 106.6652, 2.5, 55),
    ("Binh Chanh", 10.6874, 106.5933, 4.0, 60),
    ("Hoc Mon", 10.8830, 106.5920, 3.5, 50),
]
CONJ_SOURCES = [
    ("Go Vap", 10.8386, 106.6652, 2.5, 75),
    ("Tan Phu", 10.7900, 106.6280, 2.5, 70),
    ("Binh Thanh", 10.8106, 106.7091, 2.5, 50),
    ("District 8", 10.7240, 106.6286, 2.5, 55),
]
DENGUE_BASE, CONJ_BASE = 8.0, 4.0

# approximate area centres, used only to NAME the area the user is in (nearest centre)
AREAS: dict[str, LatLng] = {
    "District 1": (10.7769, 106.7009), "District 3": (10.7843, 106.6844), "District 4": (10.7578, 106.7013),
    "District 5": (10.7540, 106.6633), "District 6": (10.7467, 106.6350), "District 7": (10.7340, 106.7220),
    "District 8": (10.7240, 106.6286), "District 10": (10.7746, 106.6679), "District 11": (10.7629, 106.6500),
    "District 12": (10.8672, 106.6414), "Binh Thanh": (10.8106, 106.7091), "Go Vap": (10.8386, 106.6652),
    "Phu Nhuan": (10.7992, 106.6803), "Tan Binh": (10.8014, 106.6527), "Tan Phu": (10.7900, 106.6280),
    "Binh Tan": (10.7654, 106.6038), "Thu Duc": (10.8494, 106.7537), "Binh Chanh": (10.6874, 106.5933),
    "Hoc Mon": (10.8830, 106.5920), "Nha Be": (10.6950, 106.7380),
}


@dataclass
class Scenario:
    """User-adjustable scenario knobs. live != None replaces the simulated AQI with real data."""
    aqi_scale: float = 1.0
    dengue_scale: float = 1.0
    conj_scale: float = 1.0
    live: "LiveAQI | None" = None


def _gauss_sum(la, ln, sources) -> np.ndarray:
    la, ln = np.asarray(la, float), np.asarray(ln, float)
    out = np.zeros(np.broadcast(la, ln).shape, dtype=float)
    for _, lat0, lng0, sigma, amp in sources:
        kx = 111.32 * math.cos(math.radians(lat0))
        dx, dy = (ln - lng0) * kx, (la - lat0) * 110.57
        out = out + amp * np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    return out


def aqi_tod(minute: float) -> float:
    m = minute % 1440
    h = int(m // 60)
    frac = (m - 60 * h) / 60
    return _AQI_TOD[h] + (_AQI_TOD[(h + 1) % 24] - _AQI_TOD[h]) * frac


# ---------------------------------------------------------------- live AQI (optional, real)
@dataclass
class LiveAQI:
    pts: np.ndarray    # (N, 2) lat, lng
    vals: np.ndarray   # (N,) US AQI
    fetched_at: str


def fetch_live_aqi(bbox=GEO_BBOX, n: int = 5, session=None, timeout: float = 15.0) -> LiveAQI:
    """Current US AQI on an n x n grid from Open-Meteo's air-quality API (no key needed; CAMS model, ~10 km).

    Raises on any failure so the caller can fall back to the simulation.
    """
    s, w, nn, e = bbox
    lats, lngs = np.linspace(s, nn, n), np.linspace(w, e, n)
    pts = [(float(a), float(b)) for a in lats for b in lngs]
    params = {
        "latitude": ",".join(f"{p[0]:.4f}" for p in pts),
        "longitude": ",".join(f"{p[1]:.4f}" for p in pts),
        "current": "us_aqi",
        "timezone": "Asia/Bangkok",
    }
    r = (session or requests).get("https://air-quality-api.open-meteo.com/v1/air-quality", params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    items = js if isinstance(js, list) else [js]
    keep_pts, vals = [], []
    for p, it in zip(pts, items):
        v = (it.get("current") or {}).get("us_aqi")
        if v is not None:
            keep_pts.append(p)
            vals.append(float(v))
    if len(vals) < 4:
        raise RuntimeError("Open-Meteo returned too few AQI values")
    return LiveAQI(np.array(keep_pts), np.array(vals), datetime.now().strftime("%H:%M"))


def _idw(la, ln, pts: np.ndarray, vals: np.ndarray, power: float = 2.0) -> np.ndarray:
    la, ln = np.asarray(la, float), np.asarray(ln, float)
    num = np.zeros(np.broadcast(la, ln).shape)
    den = np.zeros_like(num)
    for (plat, plng), v in zip(pts, vals):
        kx = 111.32 * math.cos(math.radians(plat))
        d2 = ((ln - plng) * kx) ** 2 + ((la - plat) * 110.57) ** 2
        w = 1.0 / (d2 + 1.0) ** (power / 2)
        num = num + w * v
        den = den + w
    return num / den


# ---------------------------------------------------------------- fields
def field(layer: str, la, ln, minute: float, sc: Scenario) -> np.ndarray:
    """Value of the layer at (la, ln); la/ln may be scalars or arrays."""
    if layer == "aqi":
        if sc.live is not None:
            return np.clip(_idw(la, ln, sc.live.pts, sc.live.vals), 0, 500)
        return np.clip((AQI_BASE + _gauss_sum(la, ln, AQI_SOURCES)) * aqi_tod(minute) * sc.aqi_scale, 0, 500)
    if layer == "dengue":
        return np.clip((DENGUE_BASE + _gauss_sum(la, ln, DENGUE_SOURCES)) * sc.dengue_scale, 0, 100)
    if layer == "conj":
        return np.clip((CONJ_BASE + _gauss_sum(la, ln, CONJ_SOURCES)) * sc.conj_scale, 0, 100)
    raise ValueError(layer)


def value_at(layer: str, lat: float, lng: float, minute: float, sc: Scenario) -> float:
    return float(field(layer, lat, lng, minute, sc))


def make_grid(bbox=GEO_BBOX, ny: int = 100, nx: int = 110) -> tuple[np.ndarray, np.ndarray]:
    """Lat/lng grids; row 0 is the NORTH edge (the orientation an image overlay expects)."""
    s, w, n, e = bbox
    return np.meshgrid(np.linspace(n, s, ny), np.linspace(w, e, nx), indexing="ij")


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def colorize(layer: str, values: np.ndarray, alpha: int = 150) -> np.ndarray:
    """(ny, nx) values -> (ny, nx, 4) uint8 RGBA using the level colours."""
    if layer == "aqi":
        idx = np.digitize(values, [lv["max"] for lv in AQI_LEVELS[:-1]], right=True)
        pal = [_hex_rgb(lv["color"]) for lv in AQI_LEVELS]
    else:
        idx = np.digitize(values, [lv["min"] for lv in RISK_LEVELS[1:]], right=False)
        pal = [_hex_rgb(lv["color"]) for lv in RISK_LEVELS]
    rgb = np.array(pal, dtype=np.uint8)[idx]
    a = np.full(values.shape + (1,), alpha, dtype=np.uint8)
    return np.concatenate([rgb, a], axis=-1)


def nearest_area(lat: float, lng: float) -> str:
    return min(AREAS, key=lambda k: core.haversine_km((lat, lng), AREAS[k]))


def area_ranking(layer: str, minute: float, sc: Scenario):
    """Value of the layer at every area centre, worst first (list of dicts)."""
    rows = []
    for name, (la, ln) in AREAS.items():
        v = value_at(layer, la, ln, minute, sc)
        lv = level_of(layer, v)
        rows.append({"Area": name, "Value": round(v), "Level": lv["short"] if layer == "aqi" else lv["name"]})
    return sorted(rows, key=lambda r: -r["Value"])


# ====================================================================================
# 3. Hazard assessment -> basket
# ====================================================================================
PRODUCTS = {
    "n95": "N95 / KN95 masks",
    "saline_eye": "Saline (0.9% NaCl) eye drops",
    "saline_nasal": "Saline nasal spray / rinse",
    "repellent": "Mosquito repellent (DEET / picaridin)",
    "paracetamol": "Paracetamol (fever relief - follow the label)",
    "ors": "Oral rehydration salts",
    "thermometer": "Digital thermometer",
    "sanitizer": "Hand sanitizer / antiseptic wipes",
}
HAZARD_BASKETS = {  # first two items of each list are the "essential" ones
    "aqi": ["n95", "saline_eye", "saline_nasal"],
    "dengue": ["repellent", "paracetamol", "ors", "thermometer"],
    "conj": ["saline_eye", "sanitizer"],
}
ESSENTIAL_PER_HAZARD = 2

ADVICE = {
    "aqi": {
        2: "Children, older adults and people with asthma or heart/lung disease should limit long outdoor activity. "
           "Wear a well-fitted N95/KN95 mask outdoors.",
        3: "Everyone should reduce outdoor exertion. Wear an N95/KN95 outdoors and keep windows closed.",
    },
    "dengue": {
        1: "Aedes mosquitoes bite in the daytime too: use repellent and remove standing water around the home.",
        "default": "Use repellent and remove standing water. For fever lasting over 2 days, severe headache, any bleeding or "
                   "severe abdominal pain, see a doctor promptly. Avoid aspirin/ibuprofen unless a doctor advises it.",
    },
    "conj": {
        1: "Pink eye spreads by touch: wash hands, do not share towels, avoid rubbing your eyes.",
        "default": "Wash hands often and do not share towels. Saline can soothe irritation; antibiotic or steroid eye drops "
                   "should only be used on a doctor's advice. See an eye doctor for pain, light sensitivity or blurred vision.",
    },
}
DISCLAIMER = "General consumer information only, not medical advice. If you feel seriously unwell, see a doctor or call 115."


@dataclass
class Hazard:
    key: str
    label: str
    value: float
    level_name: str
    short: str
    color: str
    level_idx: int
    active: bool
    pressure: float   # 0..1 demand pressure (drives simulated stock-outs)
    advice: str


def assess(lat: float, lng: float, minute: float, sc: Scenario) -> list[Hazard]:
    out = []
    for key in LAYERS:
        v = value_at(key, lat, lng, minute, sc)
        lv = level_of(key, v)
        if key == "aqi":
            active = lv["idx"] >= AQI_ACTIVE_IDX
            pressure = float(np.clip((v - 50) / 150, 0, 1))
            adv = ADVICE["aqi"].get(min(lv["idx"], 3), "") if active else ""
        else:
            active = lv["idx"] >= RISK_ACTIVE_IDX
            pressure = float(np.clip(v / 100, 0, 1))
            tbl = ADVICE[key]
            adv = (tbl.get(lv["idx"]) or tbl["default"]) if active else ""
        out.append(Hazard(key, LAYERS[key], v, lv["name"], lv["short"], lv["color"], lv["idx"], active, pressure, adv))
    return out


def build_basket(hazards: list[Hazard]) -> list[dict]:
    """Merge the baskets of all ACTIVE hazards, most severe first, de-duplicated."""
    basket: dict[str, dict] = {}
    for h in sorted((h for h in hazards if h.active), key=lambda h: -h.pressure):
        for i, key in enumerate(HAZARD_BASKETS[h.key]):
            if key not in basket:
                basket[key] = dict(key=key, name=PRODUCTS[key], essential=i < ESSENTIAL_PER_HAZARD,
                                   pressure=h.pressure, reason=h.label)
            else:
                basket[key]["essential"] |= i < ESSENTIAL_PER_HAZARD
                basket[key]["pressure"] = max(basket[key]["pressure"], h.pressure)
    return list(basket.values())


# ====================================================================================
# 4. Stock (SIMULATED unless the CSV has stock_<product> columns) and store ranking
# ====================================================================================
def stock_status(row: dict, key: str, pressure: float) -> tuple[bool, bool]:
    """(in_stock, is_simulated). Uses a `stock_<key>` CSV column when present, else a deterministic simulation
    where high local demand (pressure) makes stock-outs more likely."""
    col = f"stock_{key}"
    if col in row and str(row[col]).strip() != "":
        return core._to_bool(row[col]), False
    h = zlib.crc32(f"{row['id']}|{key}".encode()) / 2 ** 32
    return h < 0.88 - 0.38 * pressure, True


@dataclass
class StoreOption:
    store: dict
    eta_min: float
    km: float
    arrival_min: float
    items: list[tuple[str, str, bool]]   # (key, name, in_stock)
    missing_essential: int
    missing_other: int
    cost: float
    simulated_stock: bool = True


def recommend_stores(stores, loc: LatLng, minute: float, mode: str, basket: list[dict], router=None,
                     top_k: int = 3, pool: int = 12, store_dwell: float = 8.0):
    """Open-at-arrival stores nearest to `loc`, re-ranked by missing basket items.

    cost = travel minutes + 6 per missing essential item + 2 per missing other item.
    Returns (options, router_name, fell_back).
    """
    heur = core.HeuristicRouter()
    cands = []
    for r in stores.to_dict("records"):
        s = (r["lat"], r["lng"])
        eta = heur.minutes(loc, s, mode, minute)
        if core.is_open_at(r, minute + eta, store_dwell):
            cands.append((eta, r))
    cands.sort(key=lambda t: t[0])
    cands = cands[:pool]
    router_name, fell_back, rt = heur.name, False, heur
    if router is not None and getattr(router, "online", False) and cands:
        router.prefetch([loc] + [(r["lat"], r["lng"]) for _, r in cands], mode)
        if router.ok:
            rt, router_name = router, router.name
        else:
            fell_back = True
    opts = []
    for _, r in cands:
        s = (r["lat"], r["lng"])
        eta = rt.minutes(loc, s, mode, minute)
        items, sim = [], False
        for b in basket:
            ok, is_sim = stock_status(r, b["key"], b["pressure"])
            items.append((b["key"], b["name"], ok))
            sim |= is_sim
        me = sum(1 for (k, _, ok), b in zip(items, basket) if not ok and b["essential"])
        mo = sum(1 for (k, _, ok), b in zip(items, basket) if not ok and not b["essential"])
        opts.append(StoreOption(r, eta, rt.km(loc, s), minute + eta, items, me, mo, eta + 6 * me + 2 * mo, sim))
    opts.sort(key=lambda o: o.cost)
    return opts[:top_k], router_name, fell_back


def fmt_dist(km: float) -> str:
    return f"{km * 1000:.0f} m" if km < 1 else f"{km:.1f} km"


def walk_minutes(km: float) -> float:
    return km / 5.0 * 60


def headline(area: str, hazards: list[Hazard], options: list[StoreOption]) -> str:
    """One-sentence recommendation in the style of the original idea."""
    active = [h for h in hazards if h.active]
    if not options:
        return f"No store is open when you could get there from **{area}**. Try another time."
    best = options[0]
    name, dist = best.store["name"].replace("[DEMO] ", ""), fmt_dist(best.km)
    if not active:
        return (f"No elevated environmental hazard detected around **{area}**. "
                f"The nearest open store is **{name}**, {dist} away.")
    zones = []
    for h in sorted(active, key=lambda h: -h.pressure):
        if h.key == "aqi":
            zones.append(f"a **{h.short}**-level fine-dust zone (AQI {h.value:.0f})")
        elif h.key == "dengue":
            zones.append(f"a **{h.level_name.lower()}**-risk dengue zone")
        else:
            zones.append(f"a **{h.level_name.lower()}**-risk conjunctivitis zone")
    zone_txt = " and ".join(zones)
    have = [_lc(n) for _, n, ok in best.items if ok]
    miss = [_lc(n) for _, n, ok in best.items if not ok]
    txt = f"You are in {zone_txt} (**{area}**). Head to **{name}**, {dist} away, for {_join(have)} (in stock)." if have else \
        f"You are in {zone_txt} (**{area}**). The nearest open store is **{name}**, {dist} away, but none of the recommended items is in stock there."
    if miss and have:
        txt += f" Not in stock there: {_join(miss)}."
    return txt


def _lc(name: str) -> str:
    """Lower-case the first letter for mid-sentence use, but keep acronyms like N95 / KN95."""
    return name[0].lower() + name[1:] if len(name) > 1 and name[1].islower() else name


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]
