"""
=======================================================================
STEP 3B - GPS TAGGER: Tag Detections with GPS Coordinates
=======================================================================
Supports:
  • Real GPS via gpsd daemon (Linux/Raspberry Pi with GPS USB dongle)
  • Simulated GPS for testing (follows a realistic road path)
  • NMEA sentence parsing (for raw GPS serial data)
  • Geocoding (coordinates → address) using OpenStreetMap
=======================================================================
"""

import math
import time
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


@dataclass
class GPSPoint:
    """Represents a GPS coordinate with metadata."""
    lat:       float  # Latitude  (degrees)
    lon:       float  # Longitude (degrees)
    alt:       float  # Altitude  (metres)
    speed:     float  # Speed     (km/h)
    heading:   float  # Heading   (degrees from North, 0–360)
    accuracy:  float  # Accuracy  (metres, lower is better)
    timestamp: str    # ISO 8601 UTC timestamp
    source:    str    # "gps", "network", or "simulated"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_geojson(self) -> dict:
        """Return as GeoJSON Point feature."""
        return {
            "type":       "Feature",
            "geometry":   {"type": "Point", "coordinates": [self.lon, self.lat, self.alt]},
            "properties": {
                "speed":    self.speed,
                "heading":  self.heading,
                "accuracy": self.accuracy,
                "time":     self.timestamp,
            },
        }


class GPSSimulator:
    """
    Simulates realistic GPS movement along a road for testing purposes.
    Starts at a base position and moves forward at configured speed.
    """

    def __init__(
        self,
        start_lat:   float = 12.97160,   # Default: Bangalore, India
        start_lon:   float = 77.59460,
        speed_kmh:   float = 30.0,       # Simulated vehicle speed
        heading_deg: float = 90.0,       # East
        noise_m:     float = 3.0,        # GPS noise (metres)
    ):
        self.lat     = start_lat
        self.lon     = start_lon
        self.speed   = speed_kmh
        self.heading = heading_deg
        self.noise_m = noise_m
        self._start  = time.time()

        # Precompute Earth radius in degrees-per-metre
        self._m_per_deg_lat = 111320.0
        self._m_per_deg_lon = 111320.0 * math.cos(math.radians(start_lat))

    def get(self) -> GPSPoint:
        """Return the current simulated GPS position."""
        elapsed = time.time() - self._start

        # Distance moved (metres)
        dist_m = (self.speed / 3.6) * elapsed

        # Heading in radians
        heading_rad = math.radians(self.heading)

        # Update position
        lat = self.lat + (dist_m * math.cos(heading_rad)) / self._m_per_deg_lat
        lon = self.lon + (dist_m * math.sin(heading_rad)) / self._m_per_deg_lon

        # Add realistic GPS noise
        lat += random.gauss(0, self.noise_m / self._m_per_deg_lat)
        lon += random.gauss(0, self.noise_m / self._m_per_deg_lon)

        # Simulate slight heading variation (road isn't perfectly straight)
        jitter_heading = self.heading + random.gauss(0, 2.0)

        return GPSPoint(
            lat       = round(lat, 7),
            lon       = round(lon, 7),
            alt       = 920.0 + random.gauss(0, 0.5),  # Bangalore altitude ~920m
            speed     = self.speed + random.gauss(0, 2.0),
            heading   = round(jitter_heading % 360, 1),
            accuracy  = round(abs(random.gauss(3.0, 1.0)), 1),
            timestamp = datetime.now(timezone.utc).isoformat(),
            source    = "simulated",
        )


class RealGPS:
    """
    Interface to a real GPS device via the gpsd daemon.

    Hardware Requirements:
      • Any USB GPS dongle (e.g., GlobalSat BU-353, u-blox, Adafruit Ultimate GPS)
      • Raspberry Pi or Linux PC with gpsd installed

    Setup:
      sudo apt install gpsd gpsd-clients python3-gps
      sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock
      cgps            ← test that GPS is receiving satellites
    """

    def __init__(self):
        self._connected = False
        try:
            import gpsd
            gpsd.connect()
            self._gpsd      = gpsd
            self._connected = True
            print("✅ Real GPS connected via gpsd")
        except Exception as e:
            print(f"⚠️  gpsd not available ({e}). Falling back to simulated GPS.")

    def get(self) -> GPSPoint:
        if not self._connected:
            return None
        try:
            packet = self._gpsd.get_current()
            return GPSPoint(
                lat       = round(packet.lat, 7),
                lon       = round(packet.lon, 7),
                alt       = round(packet.alt, 1) if packet.alt else 0.0,
                speed     = round(packet.speed() * 3.6, 1),   # m/s → km/h
                heading   = round(packet.movement().get("track", 0.0), 1),
                accuracy  = round(packet.position_precision()[0], 1),
                timestamp = datetime.now(timezone.utc).isoformat(),
                source    = "gps",
            )
        except Exception as e:
            print(f"⚠️  GPS read error: {e}")
            return None


class GPSTagger:
    """
    High-level GPS tagger that auto-selects between real and simulated GPS.

    Usage:
        tagger = GPSTagger()
        gps    = tagger.get_current()  # Returns GPSPoint
        print(gps.lat, gps.lon)
    """

    def __init__(
        self,
        use_real_gps: bool = False,
        start_lat:    float = 12.97160,
        start_lon:    float = 77.59460,
    ):
        self._simulator = GPSSimulator(start_lat, start_lon)
        self._real_gps  = None
        self._use_real  = use_real_gps

        if use_real_gps:
            self._real_gps = RealGPS()
            if not self._real_gps._connected:
                self._use_real = False
                print("   Using simulated GPS as fallback.")

        self._history       = []
        self._log_path      = Path(__file__).parent.parent / "gps_log.json"

    def get_current(self) -> dict:
        """Get current GPS position as a plain dict (JSON-serialisable)."""
        if self._use_real and self._real_gps:
            point = self._real_gps.get()
            if point:
                self._history.append(point.to_dict())
                return point.to_dict()

        # ── Fall back to simulator ─────────────────────────────────────
        point = self._simulator.get()
        self._history.append(point.to_dict())
        return point.to_dict()

    def get_track(self) -> list[dict]:
        """Return the full GPS track (all logged positions) as a list."""
        return self._history

    def save_track(self):
        """Save the GPS track to a GeoJSON file."""
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        for point_dict in self._history:
            geojson["features"].append({
                "type":     "Feature",
                "geometry": {
                    "type":        "Point",
                    "coordinates": [point_dict["lon"], point_dict["lat"]],
                },
                "properties": point_dict,
            })

        with open(self._log_path, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"📍 GPS track saved: {self._log_path}")

    @staticmethod
    def haversine_distance(p1: dict, p2: dict) -> float:
        """
        Calculate distance between two GPS points using the Haversine formula.

        Args:
            p1, p2: dicts with 'lat' and 'lon' keys

        Returns:
            Distance in metres
        """
        R    = 6371000   # Earth radius in metres
        lat1 = math.radians(p1["lat"])
        lat2 = math.radians(p2["lat"])
        dlat = math.radians(p2["lat"] - p1["lat"])
        dlon = math.radians(p2["lon"] - p1["lon"])

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)

    @staticmethod
    def reverse_geocode(lat: float, lon: float) -> str:
        """
        Convert GPS coordinates to a human-readable address
        using OpenStreetMap Nominatim (free, no API key needed).

        Be respectful: max 1 request/second per OSM usage policy.

        Returns:
            "Street Name, City, State" or "Unknown Location"
        """
        import requests
        try:
            url     = "https://nominatim.openstreetmap.org/reverse"
            params  = {"lat": lat, "lon": lon, "format": "json"}
            headers = {"User-Agent": "RoadQualityMonitor/1.0"}
            resp    = requests.get(url, params=params, headers=headers, timeout=5)
            data    = resp.json()
            return data.get("display_name", "Unknown Location")
        except Exception:
            return "Unknown Location"


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 Testing GPS tagger...\n")
    tagger = GPSTagger(start_lat=12.97160, start_lon=77.59460)

    for i in range(5):
        gps = tagger.get_current()
        print(f"  [{i+1}] Lat={gps['lat']:.6f}  Lon={gps['lon']:.6f}  "
              f"Speed={gps['speed']:.1f} km/h  Source={gps['source']}")
        time.sleep(0.5)

    tagger.save_track()

    # Test distance
    p1 = {"lat": 12.97160, "lon": 77.59460}
    p2 = {"lat": 12.97200, "lon": 77.59500}
    d  = GPSTagger.haversine_distance(p1, p2)
    print(f"\n📏 Distance between p1 and p2: {d} metres")
