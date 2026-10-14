"""viz.py - Folium maps (kept separate from Streamlit so it is easy to test)."""
from __future__ import annotations

import folium
import numpy as np

import core

LAMP = "#F2B134"     # 24/7 store (lights still on)
OPEN = "#2E8B6E"     # open
CLOSED = "#8A949B"   # closed
ROUTE = "#1F6FEB"
ALERT = "#D1495B"

# Night mode: OpenStreetMap has no official dark style, so we invert the standard tiles with CSS.
_TILE_CSS = """
<style>
.osm-night { filter: invert(1) hue-rotate(180deg) brightness(.92) contrast(.9) saturate(.8); }
</style>
"""


def _pin(text: str, bg: str, fg: str = "#fff", size: int = 26) -> folium.DivIcon:
    html = (
        f'<div style="width:{size}px;height:{size}px;line-height:{size - 4}px;border-radius:50%;'
        f"background:{bg};color:{fg};text-align:center;font:600 12px 'Be Vietnam Pro',sans-serif;"
        f'border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.45)">{text}</div>'
    )
    return folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size // 2, size // 2))


def _base_map(lats, lngs, night: bool) -> folium.Map:
    """Real street map. Uses the keyless OpenStreetMap tile server (CARTO basemaps now require an API key)."""
    m = folium.Map(
        location=[float(np.mean(lats)), float(np.mean(lngs))],
        zoom_start=12,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="OpenStreetMap",
        max_zoom=19,
        class_name="osm-night" if night else "osm-day",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri Streets (alternative)",
        max_zoom=19,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri, Maxar, Earthstar Geographics",
        name="Satellite (Esri)",
        max_zoom=19,
        show=False,
    ).add_to(m)
    m.get_root().header.add_child(folium.Element(_TILE_CSS))
    m.fit_bounds([[float(min(lats)) - 0.008, float(min(lngs)) - 0.008],
                  [float(max(lats)) + 0.008, float(max(lngs)) + 0.008]])
    return m


def _store_tip(row) -> str:
    area = f"{row['area']}<br>" if row.get("area") else ""
    return f"{row['name']}<br>{area}Hours: {core.hours_label(row)}"


def result_map(stores, open_now, q, rec, poly, base_poly, tol_km: float, show_hotspots: bool) -> folium.Map:
    """Result map: stores by open/closed state, top-K numbered, route, acceptable-radius circle."""
    night = core.night_weight(q.now_min) >= 0.5
    pts = [q.origin, q.dest, *q.via] + [(p.store["lat"], p.store["lng"]) for p in rec.plans]
    m = _base_map([p[0] for p in pts], [p[1] for p in pts], night)

    if show_hotspots:
        for name, la, ln, rad, _ in core.HOTSPOTS:
            folium.Circle((la, ln), radius=rad * 1000, color=ALERT, weight=1, fill=True,
                          fill_opacity=0.10, tooltip=f"Congestion hotspot (assumed): {name}").add_to(m)

    # willingness-to-travel circle (road km -> straight-line km)
    folium.Circle(q.origin, radius=tol_km / core.CIRCUITY * 1000, color="#9AA5AB", weight=1.5,
                  dash_array="6 6", fill=False,
                  tooltip=f"Acceptable radius: {tol_km:.1f} km by road").add_to(m)

    top_ids = {p.store["id"] for p in rec.plans}
    for i, row in enumerate(stores.to_dict("records")):
        if row["id"] in top_ids:
            continue
        if row["is_24h"]:
            color, rad, op = LAMP, 7, 0.95
        elif open_now[i]:
            color, rad, op = OPEN, 5, 0.85
        else:
            color, rad, op = CLOSED, 4, 0.45
        folium.CircleMarker((row["lat"], row["lng"]), radius=rad, color=color, weight=1,
                            fill=True, fill_color=color, fill_opacity=op, tooltip=_store_tip(row)).add_to(m)

    if base_poly and len(base_poly) > 1:
        folium.PolyLine(base_poly, color="#8A949B", weight=3, opacity=0.8, dash_array="4 8",
                        tooltip="Original route (no store stop)").add_to(m)
    if poly and len(poly) > 1:
        folium.PolyLine(poly, color=ROUTE, weight=5, opacity=0.9, tooltip="Route with store stop").add_to(m)

    for rank, p in reversed(list(enumerate(rec.plans, start=1))):
        s = p.store
        bg = ROUTE if rank == 1 else (LAMP if s["is_24h"] else OPEN)
        fg = "#fff" if rank == 1 or not s["is_24h"] else "#3a2a00"
        folium.Marker((s["lat"], s["lng"]), icon=_pin(str(rank), bg, fg, 30 if rank == 1 else 24),
                      tooltip=f"#{rank} {_store_tip(s)}").add_to(m)

    folium.Marker(q.origin, icon=_pin("A", "#12313A"), tooltip="Start").add_to(m)
    folium.Marker(q.dest, icon=_pin("B", "#12313A"), tooltip="End").add_to(m)
    for i, v in enumerate(q.via, start=1):
        folium.Marker(v, icon=_pin(f"V{i}", "#6B4FA0", size=24), tooltip=f"Stop {i}").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def heat_map(grid: np.ndarray, nearest: np.ndarray, stores, mask: np.ndarray,
             base_km: float, high_km: float, night: bool) -> folium.Map:
    """Each grid cell is coloured by its distance to the nearest OPEN store."""
    m = _base_map(grid[:, 0], grid[:, 1], night)
    for (la, ln), dist in zip(grid, nearest):
        if dist <= base_km:
            color = OPEN
        elif dist <= high_km:
            color = LAMP
        else:
            color = ALERT
        txt = "no store open" if not np.isfinite(dist) else f"{dist:.1f} km to the nearest open store"
        folium.CircleMarker((float(la), float(ln)), radius=7, stroke=False, fill=True, fill_color=color,
                            fill_opacity=0.55, tooltip=txt).add_to(m)
    for i, row in enumerate(stores.to_dict("records")):
        if mask[i]:
            folium.CircleMarker((row["lat"], row["lng"]), radius=7 if row["is_24h"] else 5, color="#12313A",
                                weight=2, fill=True, fill_color="#FFFFFF", fill_opacity=1,
                                tooltip=_store_tip(row)).add_to(m)
        else:
            folium.CircleMarker((row["lat"], row["lng"]), radius=3, color=CLOSED, weight=1, fill=True,
                                fill_color=CLOSED, fill_opacity=0.5, tooltip=_store_tip(row) + "<br>(closed)").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def _overlay_or_dots(m: folium.Map, rgba: np.ndarray, bounds, opacity: float = 0.85) -> None:
    """Heat layer as one smooth image. If the installed folium cannot build the image, fall back to coloured dots."""
    try:
        folium.raster_layers.ImageOverlay(
            image=rgba, bounds=bounds, opacity=opacity, origin="upper", interactive=False,
            cross_origin=False, zindex=2, pixelated=False, name="Heat layer",
        ).add_to(m)
        return
    except Exception:  # noqa: BLE001
        pass
    (s, w), (n, e) = bounds
    ny, nx = rgba.shape[:2]
    for i in range(0, ny, 4):
        for j in range(0, nx, 4):
            r, g, b, _ = (int(x) for x in rgba[i, j])
            folium.CircleMarker((n - (n - s) * i / (ny - 1), w + (e - w) * j / (nx - 1)), radius=6, stroke=False,
                                fill=True, fill_color=f"#{r:02x}{g:02x}{b:02x}", fill_opacity=0.5).add_to(m)


def geo_map(rgba: np.ndarray, bbox, loc, stores, options, route_poly, night: bool, radius_m: float = 500.0) -> folium.Map:
    """Geo-Health map: smooth heat layer + stores + 'YOU' marker + ranked store pins + route to the best store."""
    s, w, n, e = bbox
    m = _base_map([s, n], [w, e], night)
    _overlay_or_dots(m, rgba, [[s, w], [n, e]])

    top_ids = {o.store["id"] for o in options}
    for row in stores.to_dict("records"):
        if row["id"] in top_ids:
            continue
        folium.CircleMarker((row["lat"], row["lng"]), radius=3, color="#12313A", weight=1, fill=True,
                            fill_color="#FFFFFF", fill_opacity=0.9, tooltip=_store_tip(row)).add_to(m)

    folium.Circle(loc, radius=radius_m, color="#12313A", weight=1.5, dash_array="6 6", fill=False,
                  tooltip=f"{radius_m:.0f} m around you").add_to(m)
    if route_poly and len(route_poly) > 1:
        folium.PolyLine(route_poly, color=ROUTE, weight=5, opacity=0.9, tooltip="Route to the best store").add_to(m)
    for rank, o in reversed(list(enumerate(options, start=1))):
        st_ = o.store
        bg = ROUTE if rank == 1 else (LAMP if st_["is_24h"] else OPEN)
        fg = "#fff" if rank == 1 or not st_["is_24h"] else "#3a2a00"
        stock = ", ".join(f"{'✓' if ok else '✗'} {name}" for _, name, ok in o.items)
        folium.Marker((st_["lat"], st_["lng"]), icon=_pin(str(rank), bg, fg, 30 if rank == 1 else 24),
                      tooltip=f"#{rank} {_store_tip(st_)}<br>{stock}").add_to(m)
    folium.Marker(loc, icon=_pin("YOU", ALERT, size=34), tooltip="You are here (click the map to move)").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m
