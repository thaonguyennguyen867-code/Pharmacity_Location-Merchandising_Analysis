"""
core.py - pure logic (no Streamlit/Folium dependency) for three parts:

  1) Trip-Chaining Optimizer : insert one store into a trip chain  O -> [stops...] -> D
  2) T2M Urgency Matrix      : opening-hours filter + time-of-day ETA + urgency level
  3) Night supply analysis   : "supply-side" numbers for the Retail Theory section (Store Format)

Every model coefficient (speed, rush hours, congestion hotspots, access penalties...) is an
illustrative ASSUMPTION, collected in "1. Constants & assumptions" so you can calibrate it.
Road distances and durations can come from real roads (OSRM / OpenStreetMap) instead of the
offline estimate; `calibrate_circuity()` fits the offline estimate to real roads.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

LatLng = tuple[float, float]

# ====================================================================================
# 1. Constants & assumptions
# ====================================================================================
CIRCUITY = 1.35  # road distance / straight-line distance (HCMC urban default; see calibrate_circuity)
BASE_SPEED_KMH = {"motorbike": 28.0, "car": 24.0}  # free-flow speed, before the time-of-day multiplier
MODE_LABEL = {"motorbike": "Motorbike", "car": "Car"}


def set_circuity(value: float) -> None:
    """Override the offline road-distance factor (e.g. with a value calibrated from OSRM)."""
    global CIRCUITY
    CIRCUITY = float(min(max(value, 1.0), 2.5))


# Travel-time multiplier by hour of day (one value per hour, linear interpolation).
# >1 = slower (traffic), <1 = faster (empty night roads).
_TOD = [
    0.75, 0.70, 0.70, 0.70, 0.75, 0.90,  # 00-05
    1.20, 1.60, 1.70, 1.40, 1.20, 1.20,  # 06-11
    1.30, 1.20, 1.20, 1.25, 1.45, 1.80,  # 12-17
    1.85, 1.50, 1.25, 1.05, 0.90, 0.80,  # 18-23
]
MOTO_CONGESTION_SHARE = 0.7  # motorbikes only suffer 70% of the congestion slow-down

# Congestion hotspots (illustrative): name, lat, lng, radius km, rush-hour surcharge (car)
HOTSPOTS = [
    ("Hang Xanh Intersection", 10.8015, 106.7110, 1.2, 0.70),
    ("Bay Hien Intersection", 10.7933, 106.6527, 1.0, 0.60),
    ("Tan Son Nhat Airport Gate", 10.8150, 106.6650, 1.4, 0.70),
    ("Ben Thanh Market Roundabout", 10.7722, 106.6982, 1.0, 0.60),
    ("Thu Duc Intersection", 10.8497, 106.7727, 1.3, 0.60),
    ("Saigon Bridge Foot", 10.7985, 106.7320, 1.3, 0.70),
]
MOTO_HOTSPOT_FACTOR = 0.6

# Urgency levels. lam = weight on "arrive fast" vs "stay on my route";
# tol_mult = how much farther people are willing to go vs normal (DEMAND-SIDE ASSUMPTION - needs a survey).
URGENCY = {
    "low": dict(label="Low", lam=0.05, tol_mult=1.0, examples="Shampoo, vitamins, nothing needed right now"),
    "medium": dict(label="Medium", lam=0.50, tol_mult=1.5, examples="Cold medicine, a regular prescription running out"),
    "high": dict(label="High", lam=0.95, tol_mult=3.0, examples="High fever, acute stomach pain, medicine needed now"),
}

# Access penalties (minutes) added to a store's cost
PEN_CAR_NO_PARKING = 10.0
PEN_CAR_ONEWAY = 3.0
PEN_CAR_ALLEY = 3.0
PEN_MOTO_ONEWAY = 1.0
PEN_MOTO_ALLEY = 1.0
PEN_NIGHT_ALLEY = 4.0  # multiplied by night_weight

PLACES: dict[str, LatLng] = {  # approximate coordinates (type any address/coordinates for exact points)
    "Ben Thanh Market": (10.7725, 106.6980),
    "Bitexco Financial Tower": (10.7716, 106.7044),
    "Landmark 81": (10.7951, 106.7219),
    "Tan Son Nhat Airport": (10.8188, 106.6520),
    "Hang Xanh Intersection": (10.8015, 106.7110),
    "Bay Hien Intersection": (10.7933, 106.6527),
    "Ba Chieu Market": (10.8030, 106.6955),
    "Bach Khoa University (Ly Thuong Kiet)": (10.7731, 106.6600),
    "Cho Ray Hospital": (10.7545, 106.6600),
    "Binh Tay Market": (10.7500, 106.6495),
    "Crescent Mall (Phu My Hung)": (10.7290, 106.7190),
    "Aeon Mall Tan Phu": (10.8025, 106.6180),
    "Aeon Mall Binh Tan": (10.7436, 106.6130),
    "Vincom Quang Trung (Go Vap)": (10.8277, 106.6797),
    "Quang Trung Software Park": (10.8545, 106.6264),
    "Thu Duc Intersection": (10.8497, 106.7727),
    "Hi-Tech Park": (10.8500, 106.7900),
    "Thao Dien": (10.8040, 106.7350),
}

# ====================================================================================
# 2. Time & geometry
# ====================================================================================


def to_minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def fmt_hhmm(minute: float) -> str:
    mi = int(round(minute)) % 1440
    return f"{mi // 60:02d}:{mi % 60:02d}"


def tod_multiplier(minute: float, mode: str) -> float:
    m = minute % 1440
    h = int(m // 60)
    frac = (m - 60 * h) / 60
    a, b = _TOD[h], _TOD[(h + 1) % 24]
    base = a + (b - a) * frac
    if mode == "motorbike" and base > 1.0:
        base = 1.0 + MOTO_CONGESTION_SHARE * (base - 1.0)
    return base


def peak_weight(minute: float) -> float:
    """0..1 - how 'rush hour' it is (peaks at 08:00 and 17:45)."""
    m = minute % 1440
    return max(0.0, 1 - abs(m - 480) / 120, 1 - abs(m - 1065) / 135)


def night_weight(minute: float) -> float:
    """1 between 22:00-05:00, ramping down to 0 at 21:00 and 06:00."""
    m = minute % 1440
    if m >= 1320 or m < 300:
        return 1.0
    if m >= 1260:
        return (m - 1260) / 60
    if 300 <= m < 360:
        return (360 - m) / 60
    return 0.0


def haversine_km(a: LatLng, b: LatLng) -> float:
    r = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def haversine_np(lat1, lng1, lat2, lng2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lng2 - lng1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))


def axis_metrics(o: LatLng, d: LatLng, s: LatLng) -> tuple[float, float]:
    """(progress along the O->D axis, lateral offset km). 0 = at O, 1 = at D, <0 or >1 = outside the segment."""
    kx = 111.32 * math.cos(math.radians((o[0] + d[0]) / 2))
    ky = 110.57
    dx, dy = (d[1] - o[1]) * kx, (d[0] - o[0]) * ky
    sx, sy = (s[1] - o[1]) * kx, (s[0] - o[0]) * ky
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return 0.0, math.hypot(sx, sy)
    return (sx * dx + sy * dy) / l2, abs(sx * dy - sy * dx) / math.sqrt(l2)


def _hotspot_factor(p: LatLng, mode: str) -> float:
    best = 0.0
    for _, la, ln, rad, extra in HOTSPOTS:
        d = haversine_km(p, (la, ln))
        if d < rad:
            best = max(best, extra * (1 - d / rad))
    return best * (MOTO_HOTSPOT_FACTOR if mode == "motorbike" else 1.0)


def _segment_parts(a: LatLng, b: LatLng, mode: str, minute: float, n: int = 8) -> tuple[float, float]:
    """(free-flow minutes from straight line x CIRCUITY, hotspot congestion surcharge ratio)."""
    d = haversine_km(a, b) * CIRCUITY
    free = d / BASE_SPEED_KMH[mode] * 60
    pw = peak_weight(minute)
    if pw <= 0 or d <= 0:
        return free, 0.0
    acc = 0.0
    for i in range(n):
        t = (i + 0.5) / n
        acc += _hotspot_factor((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), mode)
    return free, acc / n * pw


# ====================================================================================
# 3. Routers: internal estimate or OSRM (real roads)
# ====================================================================================


def _r(p: LatLng) -> LatLng:
    return (round(p[0], 5), round(p[1], 5))


class HeuristicRouter:
    """Offline: straight line x CIRCUITY, times the time-of-day multiplier + hotspot surcharge."""

    name = "Internal estimate (offline)"
    online = False
    ok = True

    def prefetch(self, points, mode):  # keeps the interface compatible
        return None

    def minutes(self, a: LatLng, b: LatLng, mode: str, minute: float) -> float:
        if _r(a) == _r(b):
            return 0.0
        free, extra = _segment_parts(a, b, mode, minute)
        return free * (tod_multiplier(minute, mode) + extra)

    def km(self, a: LatLng, b: LatLng) -> float:
        if _r(a) == _r(b):
            return 0.0
        return haversine_km(a, b) * CIRCUITY

    def polyline(self, points, mode) -> list[LatLng]:
        return list(points)


@lru_cache(maxsize=128)
def _osrm_table(base_url: str, pts: tuple, timeout: float):
    """Return (durations seconds, distances metres) matrices for the points."""
    coords = ";".join(f"{lng:.5f},{lat:.5f}" for lat, lng in pts)
    r = requests.get(
        f"{base_url}/table/v1/driving/{coords}",
        params={"annotations": "duration,distance"},
        timeout=timeout,
    )
    r.raise_for_status()
    js = r.json()
    if js.get("code") != "Ok":
        raise RuntimeError(js.get("code"))
    dur = tuple(tuple(row) for row in js["durations"])
    dist = tuple(tuple(row) for row in js["distances"]) if js.get("distances") else None
    return dur, dist


@lru_cache(maxsize=128)
def _osrm_route(base_url: str, pts: tuple, timeout: float):
    coords = ";".join(f"{lng:.5f},{lat:.5f}" for lat, lng in pts)
    r = requests.get(
        f"{base_url}/route/v1/driving/{coords}",
        params={"overview": "full", "geometries": "geojson"},
        timeout=timeout,
    )
    r.raise_for_status()
    js = r.json()
    if js.get("code") != "Ok":
        raise RuntimeError(js.get("code"))
    return tuple((lat, lng) for lng, lat in js["routes"][0]["geometry"]["coordinates"])


class OSRMRouter(HeuristicRouter):
    """Uses OSRM for real road time AND distance, then applies the app's time-of-day multiplier + hotspot surcharge.

    Note: the public demo server router.project-osrm.org only has a car profile and no live traffic;
    motorbikes are approximated with the same road graph. If it is unreachable -> falls back to HeuristicRouter.
    """

    online = True
    name = "OSRM (real roads, OpenStreetMap)"
    _down_until = 0.0

    def __init__(self, base_url: str = "https://router.project-osrm.org", timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sec: dict[tuple[LatLng, LatLng], float] = {}
        self._m: dict[tuple[LatLng, LatLng], float] = {}
        self.ok = True

    def prefetch(self, points, mode):
        if time.time() < OSRMRouter._down_until:
            self.ok = False
            return
        pts: list[LatLng] = []
        for p in points:
            rp = _r(p)
            if rp not in pts:
                pts.append(rp)
        if len(pts) < 2 or len(pts) > 80:
            self.ok = False
            return
        try:
            dur, dist = _osrm_table(self.base_url, tuple(pts), self.timeout)
        except Exception:
            OSRMRouter._down_until = time.time() + 60  # avoid waiting for the timeout on every interaction
            self.ok = False
            return
        for i, a in enumerate(pts):
            for j, b in enumerate(pts):
                if dur[i][j] is not None:
                    self._sec[(a, b)] = dur[i][j]
                if dist is not None and dist[i][j] is not None:
                    self._m[(a, b)] = dist[i][j]
        self.ok = True

    def minutes(self, a, b, mode, minute):
        if _r(a) == _r(b):
            return 0.0
        sec = self._sec.get((_r(a), _r(b)))
        if sec is None:
            return super().minutes(a, b, mode, minute)
        free, extra = _segment_parts(a, b, mode, minute)
        return sec / 60 * tod_multiplier(minute, mode) + free * extra

    def km(self, a, b):
        m = self._m.get((_r(a), _r(b)))
        return m / 1000 if m is not None else super().km(a, b)

    def polyline(self, points, mode):
        if not self.ok:
            return list(points)
        try:
            return list(_osrm_route(self.base_url, tuple(_r(p) for p in points), self.timeout))
        except Exception:
            return list(points)


def calibrate_circuity(stores: pd.DataFrame, base_url: str = "https://router.project-osrm.org",
                       n: int = 12, seed: int = 7, timeout: float = 10.0) -> tuple[float, int] | None:
    """Estimate CIRCUITY from real roads: median(OSRM road km / straight-line km) over random store pairs.

    One OSRM /table call with `n` stores. Returns (circuity, number of pairs used) or None if OSRM is unreachable.
    """
    if len(stores) < 4:
        return None
    idx = np.random.default_rng(seed).choice(len(stores), size=min(n, len(stores)), replace=False)
    pts = [(float(stores["lat"].iloc[i]), float(stores["lng"].iloc[i])) for i in idx]
    try:
        _, dist = _osrm_table(base_url.rstrip("/"), tuple(_r(p) for p in pts), timeout)
    except Exception:
        return None
    if dist is None:
        return None
    ratios = []
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i == j or dist[i][j] is None:
                continue
            h = haversine_km(a, b)
            if h >= 0.8:  # very short pairs are dominated by snapping noise
                ratios.append(dist[i][j] / 1000 / h)
    if len(ratios) < 5:
        return None
    return float(min(max(np.median(ratios), 1.0), 2.5)), len(ratios)


_GEOCODE_CACHE: dict[str, LatLng] = {}


def parse_latlng(text: str) -> LatLng | None:
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*", text or "")
    if not m:
        return None
    lat, lng = float(m.group(1)), float(m.group(2))
    return (lat, lng) if 8 <= lat <= 24 and 102 <= lng <= 110 else None


def geocode(text: str) -> LatLng | None:
    """Look up an address with Nominatim (needs internet; only successful lookups are cached)."""
    key = text.strip().lower()
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{text}, Ho Chi Minh City, Vietnam", "format": "json", "limit": 1, "countrycodes": "vn"},
            headers={"User-Agent": "pharmacity-hcmc-academic-demo/1.0"},
            timeout=8,
        )
        js = r.json()
        if js:
            _GEOCODE_CACHE[key] = (float(js[0]["lat"]), float(js[0]["lon"]))
            return _GEOCODE_CACHE[key]
    except Exception:
        pass
    return None


# ====================================================================================
# 4. Store data
# ====================================================================================
REQUIRED_COLS = ["id", "name", "lat", "lng", "open_time", "close_time"]
OPTIONAL_DEFAULTS = {
    "area": "",
    "address_new": "",  # NEW-style address: house no., street, ward/commune, Ho Chi Minh City (no district level)
    "ward_new": "",
    "is_24h": False,
    "car_parking": False,
    "oneway_street": False,
    "access_type": "mat_tien",  # mat_tien (street frontage) | hem (alley)
    "format": "Standard",
    "is_demo": False,
}


def _to_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "co", "có", "x"}


def _parse_hhmm(v) -> float:
    s = str(v).strip()
    m = re.fullmatch(r"(\d{1,2})[:hH](\d{2})", s)
    if not m:
        return float("nan")
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59 or (h == 24 and mi > 0):
        return float("nan")
    return float(h * 60 + mi)


def load_stores(source) -> tuple[pd.DataFrame, list[str]]:
    """Read a store CSV -> (normalised DataFrame, list of warnings)."""
    df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")  # tolerates Excel CSV (BOM)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    warns: list[str] = []
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["open_min"] = df["open_time"].map(_parse_hhmm)
    df["close_min"] = df["close_time"].map(_parse_hhmm)
    for c in ("is_24h", "car_parking", "oneway_street", "is_demo"):
        df[c] = df[c].map(_to_bool)
    df["is_24h"] = df["is_24h"] | ((df["open_min"] == 0) & df["close_min"].isin([0, 1440]))
    df.loc[df["is_24h"], ["open_min", "close_min"]] = [0.0, 1440.0]

    bad = df["lat"].isna() | df["lng"].isna() | df["open_min"].isna() | df["close_min"].isna()
    if bad.any():
        warns.append(f"Dropped {int(bad.sum())} rows with missing/invalid coordinates or opening hours: "
                     f"{', '.join(df.loc[bad, 'id'].astype(str).head(8))}")
        df = df[~bad]

    at = df["access_type"].str.strip().str.lower().replace({"hẻm": "hem", "hem": "hem", "alley": "hem"})
    df["access_type"] = np.where(at == "hem", "hem", "mat_tien")
    df["id"] = df["id"].astype(str)
    return df.reset_index(drop=True), warns


def hours_label(row) -> str:
    if row["is_24h"]:
        return "24/7"
    return f"{fmt_hhmm(row['open_min'])}-{fmt_hhmm(row['close_min'])}"


def open_mask(df: pd.DataFrame, minute: float, dwell: float = 0.0) -> np.ndarray:
    """Bool array: store is open at `minute` and stays open for at least `dwell` more minutes."""
    o = df["open_min"].to_numpy(float)
    c = df["close_min"].to_numpy(float)
    c = np.where(c <= o, c + 1440, c)  # overnight stores (e.g. 06:00-01:00)
    t = minute % 1440
    ok = ((o <= t) & (t + dwell <= c)) | ((o <= t + 1440) & (t + 1440 + dwell <= c))
    return np.where(df["is_24h"].to_numpy(bool), True, ok)


def is_open_at(row: dict, minute: float, dwell: float = 0.0) -> bool:
    if row["is_24h"]:
        return True
    o, c = float(row["open_min"]), float(row["close_min"])
    if c <= o:
        c += 1440
    t = minute % 1440
    return any(o <= tt and tt + dwell <= c for tt in (t, t + 1440))


def minutes_to_close(row: dict, minute: float) -> float | None:
    if row["is_24h"]:
        return None
    o, c = float(row["open_min"]), float(row["close_min"])
    if c <= o:
        c += 1440
    t = minute % 1440
    for tt in (t, t + 1440):
        if o <= tt < c:
            return c - tt
    return None


# ====================================================================================
# 5. Trip-Chaining + T2M: store selection
# ====================================================================================


@dataclass
class Query:
    origin: LatLng
    dest: LatLng
    mode: str = "motorbike"
    now_min: float = 480.0
    urgency: str = "low"
    via: list[LatLng] = field(default_factory=list)
    via_dwell: float = 15.0
    store_dwell: float = 8.0
    require_parking: bool = False


@dataclass
class Plan:
    store: dict
    insert_pos: int  # insert after point number insert_pos of the original chain
    eta_min: float  # TRAVEL minutes from departure to the store
    arrival_min: float  # arrival clock time (minute of day, may exceed 1440)
    detour_min: float  # extra travel minutes vs. driving the original chain directly
    penalty_min: float  # access penalty (parking, one-way, alley at night...)
    cost: float
    mins_to_close: float | None
    straight_km: float
    progress: float
    lateral_km: float
    notes: list[str]
    waypoints: list[LatLng]
    eta_km: float = 0.0  # road kilometres from departure to the store (real roads when OSRM is on)


@dataclass
class Recommendation:
    plans: list[Plan]
    excluded: list[tuple[str, str]]
    chain: list[LatLng]
    router_name: str
    fell_back: bool


def access_penalty(store: dict, mode: str, minute: float) -> tuple[float, list[str]]:
    pen, notes = 0.0, []
    alley = store["access_type"] == "hem"
    if mode == "car":
        if not store["car_parking"]:
            pen += PEN_CAR_NO_PARKING
            notes.append(f"No dedicated car parking (+{PEN_CAR_NO_PARKING:.0f} min to find a spot)")
        if store["oneway_street"]:
            pen += PEN_CAR_ONEWAY
            notes.append(f"On a one-way street (+{PEN_CAR_ONEWAY:.0f} min)")
        if alley:
            pen += PEN_CAR_ALLEY
            notes.append(f"Store is in an alley (+{PEN_CAR_ALLEY:.0f} min)")
    else:
        if store["oneway_street"]:
            pen += PEN_MOTO_ONEWAY
            notes.append(f"On a one-way street (+{PEN_MOTO_ONEWAY:.0f} min)")
        if alley:
            pen += PEN_MOTO_ALLEY
            notes.append(f"Store is in an alley (+{PEN_MOTO_ALLEY:.0f} min)")
    nw = night_weight(minute)
    if alley and nw > 0:
        pen += PEN_NIGHT_ALLEY * nw
        notes.append("Alley at night: quiet and poorly visible")
    return pen, notes


def _run_chain(router, seq: list[LatLng], q: Query, store_pos: int | None = None,
               store_dwell: float | None = None) -> dict:
    """Run along `seq` from q.now_min; add dwell time at mid-trip stops and at the store."""
    dwell = q.store_dwell if store_dwell is None else store_dwell
    t = q.now_min
    travel = 0.0
    km = 0.0
    arr_store = travel_to_store = km_to_store = None
    last = len(seq) - 1
    for i in range(last):
        leg = router.minutes(seq[i], seq[i + 1], q.mode, t)
        t += leg
        travel += leg
        km += router.km(seq[i], seq[i + 1])
        j = i + 1
        if store_pos is not None and j == store_pos:
            arr_store, travel_to_store, km_to_store = t, travel, km
            t += dwell
        elif j < last:
            t += q.via_dwell
    return {"end": t, "travel": travel, "km": km, "arr_store": arr_store,
            "travel_to_store": travel_to_store, "km_to_store": km_to_store}


def _evaluate(router, row: dict, chain: list[LatLng], q: Query, lam: float, base_travel: float):
    s = (row["lat"], row["lng"])
    best: Plan | None = None
    for pos in range(len(chain) - 1):
        seq = chain[: pos + 1] + [s] + chain[pos + 1:]
        # store_dwell=0: 'detour' only measures the extra travel, not time-shift effects (e.g. 8 minutes of
        # shopping pushing the rest of the trip into a less congested hour); shopping time is shown separately.
        res = _run_chain(router, seq, q, store_pos=pos + 1, store_dwell=0.0)
        arr = res["arr_store"]
        if not is_open_at(row, arr, q.store_dwell):
            continue
        detour = max(0.0, res["travel"] - base_travel)
        pen, notes = access_penalty(row, q.mode, arr)
        eta = res["travel_to_store"]
        cost = lam * eta + (1 - lam) * detour + pen
        if best is None or cost < best.cost:
            mtc = minutes_to_close(row, arr)
            if mtc is not None and mtc < 30:
                notes = notes + [f"Closing soon: about {mtc:.0f} min left when you arrive"]
            prog, lat = axis_metrics(q.origin, q.dest, s)
            best = Plan(
                store=row, insert_pos=pos, eta_min=eta, arrival_min=arr, detour_min=detour,
                penalty_min=pen, cost=cost, mins_to_close=mtc,
                straight_km=haversine_km(q.origin, s), progress=prog, lateral_km=lat,
                notes=notes, waypoints=seq, eta_km=res["km_to_store"],
            )
    if best is None:
        return None, "Closed (or closing) by the time you arrive"
    return best, ""


def recommend(stores: pd.DataFrame, q: Query, router=None, top_k: int = 5, pool: int = 10) -> Recommendation:
    """Rank stores by cost:  cost = lam*ETA + (1-lam)*detour + access penalty.

    - Low urgency   : lam ~ 0 -> the most "on the way" store (least deviation from O->[via]->D).
    - High urgency  : lam ~ 1 -> the store you can reach fastest from where you are, ignoring mid-trip stops.
    - A store that is closed when you ARRIVE (not when you press search) is excluded.
    Stage 1 scores every store with the internal estimate; if an online router is given, the best `pool`
    candidates are re-scored with real road times and distances.
    """
    heur = HeuristicRouter()
    chain = [q.origin, *q.via, q.dest] if q.urgency != "high" else [q.origin, q.dest]
    lam = URGENCY[q.urgency]["lam"]
    rows = stores.to_dict("records")
    excluded: list[tuple[str, str]] = []

    if q.mode == "car" and q.require_parking:
        excluded += [(r["name"], "No car parking") for r in rows if not r["car_parking"]]
        rows = [r for r in rows if r["car_parking"]]

    def run(rt, rows_):
        base = _run_chain(rt, chain, q, store_dwell=0.0)["travel"]
        ok, bad = [], []
        for r in rows_:
            p, why = _evaluate(rt, r, chain, q, lam, base)
            if p:
                ok.append(p)
            else:
                bad.append((r["name"], why))
        ok.sort(key=lambda p: p.cost)
        return ok, bad

    plans, bad = run(heur, rows)
    excluded += bad
    router_name, fell_back = heur.name, False

    if router is not None and getattr(router, "online", False) and plans:
        pool_rows = [p.store for p in plans[:pool]]
        router.prefetch(chain + [(r["lat"], r["lng"]) for r in pool_rows], q.mode)
        if router.ok:
            plans2, bad2 = run(router, pool_rows)
            plans, router_name = plans2, router.name
            excluded += bad2
        else:
            fell_back = True

    return Recommendation(plans[:top_k], excluded, chain, router_name, fell_back)


# ====================================================================================
# 6. Night supply analysis (supply-side) for the Retail Theory report
# ====================================================================================


def dist_matrix(grid: np.ndarray, stores: pd.DataFrame) -> np.ndarray:
    """(G, S) estimated road distance (km) from every grid cell to every store."""
    return CIRCUITY * haversine_np(
        grid[:, [0]], grid[:, [1]], stores["lat"].to_numpy()[None, :], stores["lng"].to_numpy()[None, :]
    )


def build_grid(stores: pd.DataFrame, step_km: float = 1.0, max_km: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced 'demand' grid; keeps only cells within `max_km` of any store (approximates the urban area)."""
    lat0 = float(stores["lat"].mean())
    dlat = step_km / 110.57
    dlng = step_km / (111.32 * math.cos(math.radians(lat0)))
    lats = np.arange(stores["lat"].min() - 0.03, stores["lat"].max() + 0.03, dlat)
    lngs = np.arange(stores["lng"].min() - 0.03, stores["lng"].max() + 0.03, dlng)
    la, ln = np.meshgrid(lats, lngs, indexing="ij")
    grid = np.column_stack([la.ravel(), ln.ravel()])
    d = dist_matrix(grid, stores)
    keep = d.min(axis=1) <= max_km
    return grid[keep], d[keep]


def nearest_open_km(d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(d.shape[0], np.inf)
    return d[:, mask].min(axis=1)


def coverage_pct(nearest: np.ndarray, threshold_km: float) -> float:
    return float(np.mean(nearest <= threshold_km) * 100)


def hourly_profile(d: np.ndarray, stores: pd.DataFrame, base_km: float, mults: dict[str, float]) -> pd.DataFrame:
    rows = []
    for h in range(24):
        mask = open_mask(stores, h * 60)
        near = nearest_open_km(d, mask)
        fin = near[np.isfinite(near)]
        row = {
            "hour": h,
            "stores_open": int(mask.sum()),
            "median_km": float(np.median(fin)) if fin.size else np.nan,
            "p90_km": float(np.percentile(fin, 90)) if fin.size else np.nan,
        }
        for k, m in mults.items():
            row[f"coverage_{k}"] = coverage_pct(near, base_km * m)
        rows.append(row)
    return pd.DataFrame(rows)


def required_multiplier(near_day: np.ndarray, near_night: np.ndarray, base_km: float,
                        cap_pct: float = 90.0, max_mult: float = 10.0) -> float | None:
    """Multiplier on the acceptable radius so that night coverage equals daytime coverage at the normal radius.

    The target is capped at `cap_pct` so a few peripheral cells do not dominate the result.
    """
    target = min(coverage_pct(near_day, base_km), cap_pct)
    for m in np.arange(1.0, max_mult + 1e-9, 0.05):
        if coverage_pct(near_night, base_km * m) + 1e-9 >= target:
            return round(float(m), 2)
    return None


def summary_stats(d: np.ndarray, stores: pd.DataFrame, base_km: float, day_min: float, night_min: float) -> dict:
    md, mn = open_mask(stores, day_min), open_mask(stores, night_min)
    nd, nn = nearest_open_km(d, md), nearest_open_km(d, mn)
    med = lambda a: float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else float("nan")
    med_d, med_n = med(nd), med(nn)
    return {
        "n_total": int(len(stores)),
        "open_day": int(md.sum()),
        "open_night": int(mn.sum()),
        "share_24h": float(stores["is_24h"].mean() * 100),
        "median_day_km": med_d,
        "median_night_km": med_n,
        "median_ratio": med_n / med_d if med_d > 0 else float("nan"),
        "coverage_day": coverage_pct(nd, base_km),
        "coverage_night": coverage_pct(nn, base_km),
        "required_mult": required_multiplier(nd, nn, base_km),
    }


def catchment_expansion(d: np.ndarray, stores: pd.DataFrame, day_min: float, night_min: float,
                        step_km: float = 1.0) -> pd.DataFrame:
    """For each 24/7 store: 'nearest-store' area by day vs. by night (when other stores are closed)."""

    def counts(mask: np.ndarray) -> np.ndarray:
        c = np.zeros(len(stores), dtype=int)
        if mask.any():
            idx = np.flatnonzero(mask)
            np.add.at(c, idx[np.argmin(d[:, idx], axis=1)], 1)
        return c

    cd, cn = counts(open_mask(stores, day_min)), counts(open_mask(stores, night_min))
    out = stores.assign(area_day_km2=cd * step_km ** 2, area_night_km2=cn * step_km ** 2)
    out = out[out["is_24h"]].copy()
    out["expansion"] = np.where(out["area_day_km2"] > 0, out["area_night_km2"] / out["area_day_km2"].replace(0, np.nan), np.nan)
    return out[["id", "name", "area", "area_day_km2", "area_night_km2", "expansion"]].reset_index(drop=True)
