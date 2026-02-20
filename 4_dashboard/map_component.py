"""
=======================================================================
STEP 4A - DASHBOARD: Folium Map Component
=======================================================================
Generates an interactive Leaflet.js map (via Folium) showing:
  • Color-coded severity markers
  • Popup with damage details
  • Route polyline (GPS track)
  • Heatmap layer (optional)
  • Summary statistics panel
=======================================================================
"""

import folium
import folium.plugins as plugins
import json
import numpy as np
from pathlib import Path
from typing import Optional


# ── Severity colours (Folium uses CSS color names or hex) ──────────────
SEVERITY_ICON_MAP = {
    "HIGH":   {"color": "red",    "icon": "exclamation-sign", "prefix": "glyphicon"},
    "MEDIUM": {"color": "orange", "icon": "warning-sign",     "prefix": "glyphicon"},
    "LOW":    {"color": "green",  "icon": "info-sign",        "prefix": "glyphicon"},
}

CLASS_EMOJI = {
    "pothole": "🕳️",
    "crack":   "⚡",
    "wear":    "🌀",
    "D40":     "🕳️",
    "D00":     "⚡",
    "D10":     "⚡",
    "D20":     "⚡",
}


def build_damage_map(
    detections:    list,
    center_lat:    Optional[float] = None,
    center_lon:    Optional[float] = None,
    zoom_start:    int   = 16,
    show_heatmap:  bool  = True,
    show_route:    bool  = True,
    tile_style:    str   = "CartoDB dark_matter",
) -> folium.Map:
    """
    Build a Folium map with damage markers.

    Args:
        detections:   List of detection dicts (from detections.json)
        center_lat:   Map center latitude (auto-computed if None)
        center_lon:   Map center longitude (auto-computed if None)
        zoom_start:   Initial zoom level
        show_heatmap: Show heatmap layer
        show_route:   Show GPS route polyline
        tile_style:   Map tile theme

    Returns:
        folium.Map object
    """
    if not detections:
        # Empty map at a default location
        return folium.Map(
            location   = [center_lat or 12.971, center_lon or 77.594],
            zoom_start = zoom_start,
            tiles      = tile_style,
        )

    # ── Extract GPS points ─────────────────────────────────────────────
    lats = [d["gps"]["lat"] for d in detections if "gps" in d]
    lons = [d["gps"]["lon"] for d in detections if "gps" in d]

    if not lats:
        return folium.Map(location=[12.971, 77.594], zoom_start=zoom_start, tiles=tile_style)

    center_lat = center_lat or float(np.mean(lats))
    center_lon = center_lon or float(np.mean(lons))

    # ── Create base map ────────────────────────────────────────────────
    fmap = folium.Map(
        location         = [center_lat, center_lon],
        zoom_start       = zoom_start,
        tiles            = tile_style,
        control_scale    = True,
        prefer_canvas    = True,
    )

    # ── Layer control ──────────────────────────────────────────────────
    # Separate layers by severity so user can toggle them
    layers = {
        "HIGH":   folium.FeatureGroup(name="🔴 High Severity",   show=True),
        "MEDIUM": folium.FeatureGroup(name="🟡 Medium Severity", show=True),
        "LOW":    folium.FeatureGroup(name="🟢 Low Severity",    show=True),
    }

    # ── Add damage markers ─────────────────────────────────────────────
    for det in detections:
        gps      = det.get("gps", {})
        lat      = gps.get("lat", center_lat)
        lon      = gps.get("lon", center_lon)
        severity = det.get("severity", "LOW")
        cls      = det.get("class",    "pothole")
        conf     = det.get("confidence", 0.0)
        ts       = det.get("timestamp", "")
        bbox     = det.get("bbox", [0, 0, 0, 0])
        emoji    = CLASS_EMOJI.get(cls, "⚠️")

        icon_cfg = SEVERITY_ICON_MAP.get(severity, SEVERITY_ICON_MAP["LOW"])

        # ── Popup HTML ─────────────────────────────────────────────────
        sev_color = {"HIGH": "#dc3545", "MEDIUM": "#ffc107", "LOW": "#28a745"}.get(severity, "gray")
        popup_html = f"""
        <div style="font-family: 'Segoe UI', sans-serif; width: 240px;">
          <div style="background:{sev_color}; color:white; padding:8px 12px;
                      border-radius:6px 6px 0 0; font-weight:bold; font-size:14px;">
            {emoji} {cls.upper()} — {severity}
          </div>
          <div style="padding:10px; background:#1e1e2e; color:#cdd6f4; border-radius:0 0 6px 6px;">
            <table style="width:100%; border-collapse:collapse; font-size:12px;">
              <tr><td style="padding:3px; color:#a6adc8;">Confidence</td>
                  <td style="padding:3px; font-weight:bold;">{conf:.2%}</td></tr>
              <tr><td style="padding:3px; color:#a6adc8;">Latitude</td>
                  <td style="padding:3px;">{lat:.6f}</td></tr>
              <tr><td style="padding:3px; color:#a6adc8;">Longitude</td>
                  <td style="padding:3px;">{lon:.6f}</td></tr>
              <tr><td style="padding:3px; color:#a6adc8;">Time</td>
                  <td style="padding:3px;">{ts[11:19] if len(ts)>19 else ts}</td></tr>
              <tr><td style="padding:3px; color:#a6adc8;">Bbox</td>
                  <td style="padding:3px; font-size:10px;">{bbox}</td></tr>
            </table>
          </div>
        </div>
        """

        marker = folium.Marker(
            location = [lat, lon],
            popup    = folium.Popup(popup_html, max_width=280),
            tooltip  = f"{emoji} {cls} [{severity}] — click for details",
            icon     = folium.Icon(**icon_cfg),
        )

        layer = layers.get(severity, layers["LOW"])
        marker.add_to(layer)

    for layer in layers.values():
        layer.add_to(fmap)

    # ── GPS route polyline ─────────────────────────────────────────────
    if show_route and len(lats) > 1:
        route_coords = list(zip(lats, lons))
        route_layer  = folium.FeatureGroup(name="🛣️  GPS Route", show=True)
        folium.PolyLine(
            route_coords,
            color     = "#00d4ff",
            weight    = 3,
            opacity   = 0.7,
            tooltip   = "GPS route traveled",
        ).add_to(route_layer)

        # Start / end markers
        folium.Marker(
            [lats[0],  lons[0]],
            icon     = folium.Icon(color="blue",  icon="play",         prefix="fa"),
            tooltip  = "▶ Route Start",
        ).add_to(route_layer)
        folium.Marker(
            [lats[-1], lons[-1]],
            icon     = folium.Icon(color="black", icon="flag-checkered", prefix="fa"),
            tooltip  = "🏁 Route End",
        ).add_to(route_layer)

        route_layer.add_to(fmap)

    # ── Heatmap layer ──────────────────────────────────────────────────
    if show_heatmap and len(lats) > 2:
        # Weight HIGH severity more (3x), MEDIUM (2x), LOW (1x)
        weight_map  = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
        heat_data   = [
            [d["gps"]["lat"], d["gps"]["lon"],
             weight_map.get(d.get("severity", "LOW"), 1.0)]
            for d in detections if "gps" in d
        ]
        heat_layer  = folium.FeatureGroup(name="🌡️  Damage Heatmap", show=False)
        plugins.HeatMap(
            heat_data,
            min_opacity = 0.4,
            max_zoom    = 18,
            radius      = 25,
            blur        = 15,
            gradient    = {"0.4": "blue", "0.65": "yellow", "1": "red"},
        ).add_to(heat_layer)
        heat_layer.add_to(fmap)

    # ── Summary minimap ────────────────────────────────────────────────
    plugins.MiniMap(toggle_display=True).add_to(fmap)

    # ── Fullscreen button ──────────────────────────────────────────────
    plugins.Fullscreen(position="topright").add_to(fmap)

    # ── Measure tool ──────────────────────────────────────────────────
    plugins.MeasureControl(position="bottomleft").add_to(fmap)

    # ── Layer control panel ────────────────────────────────────────────
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

    return fmap


def get_map_statistics(detections: list) -> dict:
    """
    Compute summary statistics shown in the dashboard stats panel.

    Returns:
        dict with counts, percentages, and geographic bounds
    """
    if not detections:
        return {
            "total": 0, "high": 0, "medium": 0, "low": 0,
            "by_class": {}, "high_pct": 0, "medium_pct": 0, "low_pct": 0,
        }

    total    = len(detections)
    high     = sum(1 for d in detections if d.get("severity") == "HIGH")
    medium   = sum(1 for d in detections if d.get("severity") == "MEDIUM")
    low      = sum(1 for d in detections if d.get("severity") == "LOW")

    by_class = {}
    for d in detections:
        cls = d.get("class", "unknown")
        by_class[cls] = by_class.get(cls, 0) + 1

    lats = [d["gps"]["lat"] for d in detections if "gps" in d]
    lons = [d["gps"]["lon"] for d in detections if "gps" in d]

    avg_conf = np.mean([d.get("confidence", 0) for d in detections]) if detections else 0

    return {
        "total":      total,
        "high":       high,
        "medium":     medium,
        "low":        low,
        "by_class":   by_class,
        "high_pct":   round(high   / total * 100, 1),
        "medium_pct": round(medium / total * 100, 1),
        "low_pct":    round(low    / total * 100, 1),
        "avg_conf":   round(float(avg_conf), 3),
        "lat_range":  [min(lats), max(lats)] if lats else [0, 0],
        "lon_range":  [min(lons), max(lons)] if lons else [0, 0],
        "coverage_km": _estimate_route_length(lats, lons),
    }


def _estimate_route_length(lats: list, lons: list) -> float:
    """Estimate total route length in kilometres."""
    import math
    if len(lats) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(lats)):
        dlat = math.radians(lats[i] - lats[i-1])
        dlon = math.radians(lons[i] - lons[i-1])
        a    = (math.sin(dlat/2)**2 +
                math.cos(math.radians(lats[i-1])) *
                math.cos(math.radians(lats[i]))   *
                math.sin(dlon/2)**2)
        total += 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) * 6371
    return round(total, 3)
