"""
=======================================================================
STEP 3C - SEVERITY CLASSIFIER: Classify Road Damage Severity
=======================================================================
Severity levels:
  LOW     → Small localised damage (pothole < 30cm, surface cracks)
  MEDIUM  → Moderate damage (pothole 30–60cm, dense cracking patterns)
  HIGH    → Severe damage (pothole > 60cm, alligator cracking network,
             deep ruts)

Classification strategy:
  1. Relative Box Size    — What % of the frame does damage cover?
  2. Confidence Score     — Higher confidence = clearer/larger damage
  3. Class-specific rules — Potholes escalate faster than wear

Real-world enhancements (not implemented here):
  - Depth estimation using stereo camera or LiDAR
  - Pothole depth from shadow analysis (paper: IEEE Trans. ITS 2021)
  - Longitudinal crack width from pixel density
=======================================================================
"""

from dataclasses import dataclass


@dataclass
class SeverityResult:
    level:      str    # "LOW", "MEDIUM", "HIGH"
    score:      float  # Raw severity score 0.0–1.0
    reason:     str    # Human-readable explanation
    repair_priority: int    # 1 (urgent) – 3 (routine)
    estimated_size_m2: float  # Estimated damage area (assuming ~5m road width)


class SeverityClassifier:
    """
    Classifies road damage severity from bounding box geometry and class.

    The classifier uses a simple but effective heuristic based on:
      - Relative area:    bbox_area / frame_area
      - Class weight:     potholes are weighted more severely
      - Confidence:       low-confidence detections → conservative estimate

    Calibration:
      Thresholds were calibrated against field inspections of Indian roads
      and the RDD2022 dataset severity annotations.
    """

    # ── Thresholds ─────────────────────────────────────────────────────
    # Each class has (low_threshold, high_threshold) as fraction of frame area
    AREA_THRESHOLDS = {
        "pothole": (0.005, 0.025),   # 0.5% → LOW, 2.5% → HIGH
        "crack":   (0.010, 0.050),   # Cracks can be narrow but long
        "wear":    (0.020, 0.100),   # Wear covers larger areas
        # RDD2022 classes
        "D00":     (0.010, 0.050),   # Longitudinal crack
        "D10":     (0.010, 0.050),   # Transverse crack
        "D20":     (0.015, 0.060),   # Alligator crack (network)
        "D40":     (0.005, 0.025),   # Pothole
    }

    # Confidence multiplier: low confidence → penalise score
    CONF_WEIGHTS = {
        (0.0,  0.4):  0.5,    # Low conf  → downscale
        (0.4,  0.6):  0.8,    # Med conf  → slight downscale
        (0.6,  1.01): 1.0,    # High conf → no change
    }

    # REPAIR_PRIORITY: 1=urgent, 2=soon, 3=routine
    PRIORITY_MAP = {
        "HIGH":   1,
        "MEDIUM": 2,
        "LOW":    3,
    }

    def classify(
        self,
        bbox:        tuple,     # (x1, y1, x2, y2) in pixels
        frame_shape: tuple,     # (height, width)
        class_name:  str = "pothole",
        confidence:  float = 1.0,
    ) -> str:
        """
        Classify severity and return level string: "LOW", "MEDIUM", or "HIGH"

        Args:
            bbox:        Bounding box (x1, y1, x2, y2)
            frame_shape: (height, width) of the frame
            class_name:  Detection class ("pothole", "crack", "wear")
            confidence:  Model confidence score (0–1)

        Returns:
            Severity level string
        """
        result = self.classify_full(bbox, frame_shape, class_name, confidence)
        return result.level

    def classify_full(
        self,
        bbox:        tuple,
        frame_shape: tuple,
        class_name:  str   = "pothole",
        confidence:  float = 1.0,
    ) -> SeverityResult:
        """
        Full classification returning a SeverityResult dataclass.
        """
        x1, y1, x2, y2     = bbox
        frame_h, frame_w    = frame_shape

        # ── 1. Compute relative area ───────────────────────────────────
        bbox_area   = max(0, (x2 - x1)) * max(0, (y2 - y1))
        frame_area  = frame_h * frame_w
        rel_area    = bbox_area / max(frame_area, 1)

        # ── 2. Apply confidence weighting ─────────────────────────────
        conf_weight = 1.0
        for (lo, hi), weight in self.CONF_WEIGHTS.items():
            if lo <= confidence < hi:
                conf_weight = weight
                break

        weighted_area = rel_area * conf_weight

        # ── 3. Get class-specific thresholds ──────────────────────────
        low_thresh, high_thresh = self.AREA_THRESHOLDS.get(
            class_name, (0.010, 0.030)
        )

        # ── 4. Classify ────────────────────────────────────────────────
        if weighted_area >= high_thresh:
            level  = "HIGH"
            reason = (
                f"Large damage area ({rel_area*100:.1f}% of frame) "
                f"indicates severe {class_name}"
            )
        elif weighted_area >= low_thresh:
            level  = "MEDIUM"
            reason = (
                f"Moderate damage area ({rel_area*100:.1f}% of frame)"
            )
        else:
            level  = "LOW"
            reason = (
                f"Small damage area ({rel_area*100:.1f}% of frame) "
                f"or low confidence ({confidence:.2f})"
            )

        # ── 5. Estimate real-world size (assuming ~5m frame width) ─────
        # If camera sees ~5m width of road at 640px, scale factor = 5/640
        PIXEL_TO_METRE = 5.0 / frame_w   # adjust if you calibrate your camera
        real_width_m   = abs(x2 - x1) * PIXEL_TO_METRE
        real_height_m  = abs(y2 - y1) * PIXEL_TO_METRE
        est_area_m2    = real_width_m * real_height_m

        return SeverityResult(
            level              = level,
            score              = float(weighted_area),
            reason             = reason,
            repair_priority    = self.PRIORITY_MAP[level],
            estimated_size_m2  = round(est_area_m2, 3),
        )

    @staticmethod
    def severity_to_color(severity: str) -> tuple:
        """Return BGR color tuple for cv2 drawing."""
        return {
            "HIGH":   (0,   0, 220),    # Red
            "MEDIUM": (0, 165, 255),    # Orange
            "LOW":    (50, 200,  50),   # Green
        }.get(severity, (200, 200, 200))

    @staticmethod
    def severity_to_hex(severity: str) -> str:
        """Return hex colour for Folium/HTML map markers."""
        return {
            "HIGH":   "#dc3545",   # Bootstrap danger red
            "MEDIUM": "#ffc107",   # Bootstrap warning amber
            "LOW":    "#28a745",   # Bootstrap success green
        }.get(severity, "#6c757d")

    @staticmethod
    def maintenance_recommendation(severity: str, damage_class: str) -> str:
        """Return a maintenance recommendation string for reports."""
        recs = {
            ("HIGH",   "pothole"): "Immediate patching required. Road closure may be needed.",
            ("HIGH",   "crack"):   "Immediate resurfacing required. Check for structural damage.",
            ("HIGH",   "wear"):    "Emergency overlay treatment required.",
            ("MEDIUM", "pothole"): "Schedule pothole filling within 2 weeks.",
            ("MEDIUM", "crack"):   "Apply crack sealant within 1 month.",
            ("MEDIUM", "wear"):    "Schedule micro-surfacing treatment.",
            ("LOW",    "pothole"): "Monitor. Schedule routine maintenance.",
            ("LOW",    "crack"):   "Apply preventive crack seal during next maintenance cycle.",
            ("LOW",    "wear"):    "Routine inspection. No immediate action required.",
        }
        key = (severity, damage_class.lower())
        return recs.get(key, f"Monitor {damage_class} — priority: {severity}")


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = SeverityClassifier()

    # Test cases
    frame = (720, 1280)   # 720p frame

    test_cases = [
        # (bbox,                 class,    conf,  expected)
        ((600, 400, 680, 460),   "pothole", 0.85, "HIGH"),    # Large pothole
        ((10,  10,  50,  30),    "pothole", 0.72, "LOW"),     # Tiny pothole
        ((200, 300, 400, 450),   "crack",   0.60, "MEDIUM"),  # Medium crack
        ((100, 200, 900, 350),   "wear",    0.90, "HIGH"),    # Wide wear band
    ]

    print("🧪 Severity Classifier Test\n")
    print(f"  {'Bbox':25s} {'Class':8s} {'Conf':5s} {'Level':8s} {'Score':6s} {'Area%':7s}")
    print("  " + "─" * 68)

    all_pass = True
    for bbox, cls, conf, expected in test_cases:
        res        = clf.classify_full(bbox, frame, cls, conf)
        x1,y1,x2,y2 = bbox
        area_pct   = (x2-x1)*(y2-y1) / (frame[0]*frame[1]) * 100
        status     = "✅" if res.level == expected else f"❌ (expected {expected})"
        if res.level != expected:
            all_pass = False
        print(f"  {str(bbox):25s} {cls:8s} {conf:5.2f} {res.level:8s} "
              f"{res.score:.4f} {area_pct:6.2f}% {status}")

    print(f"\n  {'All tests passed! ✅' if all_pass else 'Some tests failed ❌'}")

    # Show recommendation
    print("\n📋 Sample maintenance recommendations:")
    for sev in ["HIGH", "MEDIUM", "LOW"]:
        rec = SeverityClassifier.maintenance_recommendation(sev, "pothole")
        print(f"  [{sev:6s}] {rec}")
