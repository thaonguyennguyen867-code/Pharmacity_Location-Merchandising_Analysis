# Pharmacity HCMC: Trip-Chaining x T2M Urgency x Geo-Health

A Python (Streamlit) web app that merges three ideas:

- **Trip-Chaining Optimizer**: enter a start point (office), an end point (home), a vehicle (motorbike / car) and up to 2 optional mid-trip stops. The app inserts a store into the trip chain where it deviates least and ranks stores by the extra minutes. Cars are penalised if a store has no parking or sits on a one-way street.
- **T2M Urgency Matrix**: a simulated-time slider plus an urgency level (Low / Medium / High). A store is excluded if it is **closed when you arrive** (not when you press search). ETA changes with the hour (rush hour, empty night roads, congestion hotspots).

- **Geo-Health Contextual Map** (new tab): a heat map of air quality (AQI), dengue risk and conjunctivitis risk. Click where you are; the app assesses each hazard at that spot, builds a recommended basket (e.g. N95 masks + saline eye drops) and ranks nearby open stores by travel time and how many basket items are in stock.

The first two ideas are tied together by one formula:

```
cost = lam x ETA + (1 - lam) x detour + access penalty        (minutes, lower is better)
lam = 0.05 (Low: stay on route) | 0.50 (Medium) | 0.95 (High: arrive fast, ignore mid-trip stops)
```

Scope: Pharmacity in Ho Chi Minh City, new-style addresses after the merger (`house no., street, ward/commune, Ho Chi Minh City`, no district level).

## What's new in this version

| Problem | Fix |
|---|---|
| Map showed "API KEY REQUIRED" watermarks and no real basemap | CARTO's `CartoDB positron/dark_matter` tiles now need an API key. The app uses the keyless **OpenStreetMap** tiles (night mode inverts them with CSS), plus Esri street/satellite as switchable layers. |
| Routes were straight lines | **Real roads (OSRM)** is now the default router: routes follow real roads and the app shows **road km** as well as minutes. It falls back to the offline estimate automatically if you are offline. |
| Store coordinates were synthetic | `osm_stores.py` downloads **real Pharmacity locations** (coordinates and, where mapped, opening hours) from OpenStreetMap. The sidebar source "Real stores (OpenStreetMap)" has a one-click download/refresh button. |
| Night-analysis distances used a guessed x1.35 road factor | The sidebar button **"Calibrate road factor from real roads"** measures the real ratio (OSRM road km / straight-line km) over a sample of store pairs and uses it. |
| Vietnamese UI | Fully English UI, code comments and docs. |

## Geo-Health Contextual Map (idea 3)

Tab **Geo-Health map**:

1. Pick a layer: **AQI**, **Dengue fever risk** or **Conjunctivitis (pink eye) risk**. The heat layer is one smooth image over the city; AQI uses the US EPA colour bands (Orange = 101-150).
2. Click on the map (or choose a landmark / type an address) to set where you are.
3. The app shows the three hazard levels at your spot and a recommendation such as: *"You are in a Purple-level fine-dust zone (Thu Duc). Head to Pharmacity X, 100 m away, for N95 / KN95 masks and saline eye drops (in stock)."*
4. A basket table shows which of the top 3 stores has each item; the ranking prefers a slightly farther store that has everything over the nearest one that does not (`cost = travel min + 6 per missing essential item + 2 per missing other item`).

| Part | Real or simulated |
|---|---|
| Store locations, opening hours | Real (OpenStreetMap) / your CSV |
| Distances and routes | Real roads (OSRM) |
| **AQI** | **Simulated** by default (baseline + illustrative hotspots x time-of-day x scenario). Optional toggle **"Use live AQI"** pulls real current data from Open-Meteo (a coarse ~10 km air-quality model, interpolated; it cannot show street-level differences). |
| **Dengue / conjunctivitis** | **Simulated scenarios.** No open real-time feed is wired in. Hotspots live in `DENGUE_SOURCES` / `CONJ_SOURCES` in `geohealth.py`; replace them with real data (e.g. HCDC bulletins by area) before drawing conclusions. |
| **Stock ("in stock")** | **Simulated** (deterministic per store and item; high local demand makes stock-outs more likely). To use real inventory, add columns such as `stock_n95`, `stock_saline_eye`, `stock_repellent` ... (yes/no) to the stores CSV; the app uses them automatically. |
| Product advice text | General consumer information, not medical advice (dengue: avoid aspirin/ibuprofen unless a doctor says so; pink eye: antibiotic/steroid drops only on a doctor's advice). |

Scenario controls (clean / typical / hazy / severe haze; low / typical / outbreak) let you show how the recommendation changes. For the Retail Theory section the interesting overlay is where a hazard layer and stock-outs coincide: that is where a store format with deeper health-context stock, or 24/7 hours, matters most.

Hazard thresholds, product baskets and advice are in `geohealth.py` (`AQI_ACTIVE_IDX`, `RISK_ACTIVE_IDX`, `HAZARD_BASKETS`, `ADVICE`).

## Run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python osm_stores.py                                   # optional: download real stores (needs internet)
streamlit run app.py
python tests/test_core.py                              # core logic tests (no streamlit needed)
python tests/test_osm_and_roads.py                     # tests for OSM import / OSRM km / calibration
python tests/test_geohealth.py                         # tests for the Geo-Health module
```

You can also click **Download / refresh from OpenStreetMap** in the app's sidebar when running locally. For a deployed app see 'Deploying' below.

## Files

| File | Role |
|---|---|
| `core.py` | All logic: opening hours, ETA, Trip-Chaining, ranking, night analysis, OSRM router, road-factor calibration. Every assumed coefficient is in section "1. Constants & assumptions". |
| `osm_stores.py` | Downloads real Pharmacity stores from OpenStreetMap (Overpass API) and writes the store CSV. |
| `geohealth.py` | Geo-Health logic: environment fields (simulated / live AQI), hazard levels, basket, stock, store ranking. |
| `viz.py` | Folium maps (search result, coverage heat map, Geo-Health heat layer) on OpenStreetMap tiles. |
| `app.py` | Streamlit UI: 4 tabs (Find a store, Night analysis, Geo-Health map, Data). |
| `data/stores_hcmc_demo.csv` | 43 **synthetic** stores for trying the app. |
| `data/stores_hcmc_osm.csv` | Created by `osm_stores.py` - real coordinates. |
| `tools/geocode_stores.py` | Turns a list of new-style addresses into a CSV with coordinates (alternative to OSM). |
| `tests/` | Logic tests. |

## Data: what is real and what is not

`osm_stores.py` queries OpenStreetMap for every element named/branded "Pharmacity" inside a bounding box (default `10.62,106.52,10.95,106.90`, the urban core; pass `--bbox south,west,north,east` to change it).

| Column | Source |
|---|---|
| `lat`, `lng`, `name` | **Real** (OpenStreetMap). Spot-check a few against Google Maps; OSM is crowd-sourced, so some stores may be missing, moved or closed. |
| `open_time`, `close_time`, `is_24h` | **Real if a mapper added `opening_hours`**, otherwise **assumed 07:30-22:00**. `hours_source` = `osm` / `assumed` tells you which. |
| `address_new`, `ward_new`, `area` | Built from OSM `addr:*` tags. Often partial and sometimes old-style; the ward is blank when untagged. |
| `access_type` | **Heuristic**: `hem` if the house number looks like `45/3` or the street contains hẻm/ngõ/kiệt. |
| `car_parking`, `oneway_street` | **Not in OSM** - default `False`. Needs a survey (Street View / site visit) if car parking or one-way matters for your argument. |

Important for the Retail Theory section: the night analysis depends almost entirely on **which stores are really 24/7**. If OSM lacks hours for the stores that matter, use the sidebar **"Override: mark these stores as 24/7"** or edit the CSV with verified hours. The Night-analysis tab warns you whenever hours are assumed.

Your own data still works: build `data/stores_raw.csv` (id, name, address_new, open_time, close_time, optional columns), run `python tools/geocode_stores.py`, then upload `stores_hcmc.csv` with "Upload your CSV".

| Column | Meaning |
|---|---|
| `id, name, lat, lng, open_time, close_time` | Required. Times as HH:MM; overnight like `06:30`-`01:00` is supported; 24/7 is `00:00`-`24:00`. |
| `address_new, ward_new, area` | New-style address, new ward/commune, area name. |
| `is_24h` | Open 24/7 (inferred automatically from 00:00-24:00). |
| `car_parking` | Has car parking. |
| `oneway_street` | Sits on a one-way street. |
| `access_type` | `mat_tien` (street frontage) or `hem` (alley). Alleys get an extra night penalty. |
| `format` | Store format (Standard, Flagship, 24/7...). |

## Deploying (Streamlit Community Cloud / GitHub)

**Do not rely on the in-app "Download from OpenStreetMap" button in the cloud.** Public Overpass servers often refuse or
rate-limit connections from cloud hosts (you will see `Connection refused` / `Read timed out`). Instead, generate the store
file once on your own computer and commit it:

1. `python osm_stores.py` (creates `data/stores_hcmc_osm.csv`). Open it and check it has rows and real coordinates.
   - If this also fails on your machine: run `python osm_stores.py --print-query`, paste the query into https://overpass-turbo.eu,
     press Run, then Export -> **raw OSM data** -> download `.json`, and run `python osm_stores.py --from-json export.json`.
2. Make sure `data/stores_hcmc_osm.csv` and `data/stores_hcmc_demo.csv` are NOT ignored by `.gitignore` (no `*.csv` or `data/` line).
3. Commit and push both files (`git add data/*.csv && git commit -m "add store data" && git push`).
4. In Streamlit Cloud open your app -> menu -> **Reboot app** (it also redeploys automatically on every push).
5. Check: the sidebar shows "Real stores (OpenStreetMap)" with no error, and the Data tab lists stores with an `osm_url` column.

Refresh the data now and then by repeating steps 1-4 locally. The repository must also contain `app.py, core.py, viz.py, osm_stores.py,
geohealth.py, requirements.txt`. The app still works if the file is missing: it falls back to the demo data and shows a notice.

## Distances: how "real" are they?

- **Real roads (OSRM)**: road time and road km come from OpenStreetMap's road network via the public demo server `router.project-osrm.org`. Caveats: it only has a *car* profile (motorbikes use the same graph) and **no live traffic**, so the app multiplies times by its own time-of-day factor and adds hotspot surcharges. For true traffic-aware times, write a Router that calls Google Distance Matrix / Goong / TomTom with `departure_time` (interface: `minutes(a, b, mode, minute)`, `km(a, b)`, `prefetch(points, mode)`, `polyline(points, mode)` in `core.py`). The public demo server is for light use only; self-host OSRM for heavy use.
- **Offline estimate**: straight line x road factor (default 1.35). Use the calibration button to replace 1.35 with a measured value. The Night-analysis tab always uses this estimate (grid x stores is too many pairs for the public OSRM server), so calibrating matters there.
- Caveat: the road factor is stored in a module-level variable, so if several people use one shared Streamlit server simultaneously they would overwrite each other's value. Fine for a local/class project.

## Assumptions to calibrate (in `core.py`)

- Base speed, time-of-day congestion multipliers (`_TOD`), congestion hotspots (`HOTSPOTS`), access penalties (`PEN_*`). These are illustrative, not calibrated with real traffic data.
- "Alley" (hẻm) from the original idea is modelled as `access_type = hem` plus a night penalty (quiet, poorly visible). Change `access_penalty()` if you interpret it differently.
- The mid-trip landmark list (`PLACES`) uses approximate coordinates; for an exact point choose "Type another address or coordinates…" (address lookup uses Nominatim and needs internet).

## Notes for the Retail Theory section

The "Night analysis" tab separates two things:

- **Provable from data (supply side)**: when most stores close, how much the distance to the nearest open store grows, how much coverage drops, how many times larger each 24/7 store's catchment becomes, and how many times farther customers must be willing to travel to keep night coverage equal to daytime coverage.
- **Only an assumption (demand side)**: the "customers travel 3x farther when urgent" multiplier in the sidebar. The app cannot prove it; it needs a customer survey or night-time transaction data. The sensible use: the app produces the *required* multiplier, and you compare it with evidence about the *actual* one.

Not in the app: per-store night revenue and operating cost (staff, security, electricity), so it answers "how much catchment does 24/7 preserve", not "is 24/7 profitable".

## What has been tested

- `tests/test_osm_and_roads.py` (15 tests, offline, network mocked): opening-hours parsing, OSM element conversion and de-duplication, CSV round-trip into `core.load_stores`, Overpass mirror fallback and time budget, manual JSON import, OSRM real-km handling, road-factor calibration.
- `tests/test_geohealth.py` (19 tests, offline): level boundaries, colour mapping, hotspot/time-of-day/scenario behaviour, live-AQI parsing and interpolation (mocked), hazard assessment, basket merge, stock logic, store ranking, recommendation text.
- `app.py` (including the Geo-Health tab: layers, click-to-locate, live AQI success/failure, landmark mode, clean area) was smoke-run through 16 scenarios (real/demo data, offline/online router with fallback, car, night, urgency high, mid-trip stop, calibration button) using Streamlit/Folium stubs to catch logic errors.
- **Not verified**: live calls to Overpass, OSRM, Open-Meteo and the OSM tile server (no internet in the build environment), and the real browser rendering of the maps (heat image overlay, click-to-locate). Please run it once and report anything odd.
