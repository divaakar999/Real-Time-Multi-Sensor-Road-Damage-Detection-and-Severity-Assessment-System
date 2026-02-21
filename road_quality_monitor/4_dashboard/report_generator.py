"""
=======================================================================
STEP 4B - DASHBOARD: PDF and CSV Report Generator
=======================================================================
Generates professional PDF and CSV reports for:
  • Road authorities
  • Field maintenance teams
  • Project presentations/documentation
=======================================================================
"""

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── fpdf2 for PDF generation ───────────────────────────────────────────
try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    print("⚠️  fpdf2 not installed. Run: pip install fpdf2")


REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ── Color constants ──────────────────────────────────────────────────
RED    = (220,  53,  69)
ORANGE = (255, 193,   7)
GREEN  = ( 40, 167,  69)
DARK   = ( 13,  17,  23)
BLUE   = ( 13, 110, 253)
WHITE  = (255, 255, 255)
LIGHT  = (248, 249, 250)
GRAY   = (108, 117, 125)


class RoadDamagePDF(FPDF):
    """Custom FPDF class with header and footer."""

    def __init__(self, title="Road Quality Inspection Report", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        # Dark header bar
        self.set_fill_color(*DARK)
        self.rect(0, 0, 210, 18, "F")
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 11)
        self.set_xy(8, 5)
        self.cell(0, 8, f"🛣️  {self.report_title}", align="L")
        self.set_xy(0, 5)
        self.set_font("Helvetica", "", 8)
        page_str = f"Page {self.page_no()}"
        self.cell(202, 8, page_str, align="R")
        self.ln(14)

    def footer(self):
        self.set_y(-12)
        self.set_fill_color(*DARK)
        self.rect(0, self.get_y(), 210, 15, "F")
        self.set_text_color(*GRAY)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, "AI Road Quality Monitoring System  |  Confidential", align="C")

    def section_title(self, text: str, color=BLUE):
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, f"  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(30, 30, 30)
        self.ln(2)

    def kpi_box(self, label: str, value: str, color: tuple, x: float, y: float, w: float = 42):
        """Draw a KPI card."""
        self.set_xy(x, y)
        self.set_fill_color(*color)
        self.rect(x, y, w, 18, "F")
        # Round rect effect (just borders)
        self.set_draw_color(*WHITE)
        self.set_line_width(0.5)
        self.rect(x, y, w, 18)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 14)
        self.set_xy(x, y + 2)
        self.cell(w, 8, value, align="C")
        self.set_font("Helvetica", "", 7)
        self.set_xy(x, y + 10)
        self.cell(w, 5, label.upper(), align="C")
        self.set_text_color(30, 30, 30)


def generate_pdf_report(
    detections:   list,
    model_metrics: Optional[dict] = None,
    route_name:   str = "Road Survey",
    inspector:    str = "AI System",
    output_path:  Optional[Path] = None,
) -> bytes:
    """
    Generate a professional PDF inspection report.

    Args:
        detections:    List of detection dicts
        model_metrics: Optional dict with mAP, precision, recall
        route_name:    Name of the surveyed road/route
        inspector:     Inspector name or agency
        output_path:   If given, also save to disk

    Returns:
        PDF as bytes (for Streamlit download button)
    """
    if not FPDF_AVAILABLE:
        raise ImportError("Install fpdf2: pip install fpdf2")

    stats     = _compute_stats(detections)
    now       = datetime.now()
    date_str  = now.strftime("%d %B %Y")
    time_str  = now.strftime("%H:%M")

    pdf = RoadDamagePDF(f"Road Quality Report — {route_name}")
    pdf.add_page()

    # ── Cover / Executive Summary ──────────────────────────────────────
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 14, 210, 40, "F")
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 20)
    pdf.cell(0, 10, "Road Quality Inspection Report")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 33)
    pdf.cell(0, 6, f"Route: {route_name}   |   Date: {date_str}   |   Inspector: {inspector}")
    pdf.ln(48)

    # ── KPI Row ────────────────────────────────────────────────────────
    pdf.section_title("Executive Summary", DARK)
    y_kpi = pdf.get_y() + 2
    pdf.kpi_box("Total Detected",    str(stats["total"]),           BLUE,   10,  y_kpi)
    pdf.kpi_box("High Severity",     str(stats["high"]),            RED,    56,  y_kpi)
    pdf.kpi_box("Medium Severity",   str(stats["medium"]),          ORANGE, 102, y_kpi)
    pdf.kpi_box("Low Severity",      str(stats["low"]),             GREEN,  148, y_kpi)
    pdf.ln(28)

    # ── Summary text ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    overall = _overall_condition(stats)
    summary = (
        f"The AI-based road quality monitoring system analyzed {route_name} "
        f"on {date_str} at {time_str}. A total of {stats['total']} road surface "
        f"defects were automatically detected and classified using a YOLOv8 deep "
        f"learning model. The overall road condition is assessed as {overall}. "
        f"Immediate maintenance attention is recommended for the {stats['high']} "
        f"high-severity locations identified in this report."
    )
    pdf.multi_cell(0, 6, summary)
    pdf.ln(4)

    # ── By Class breakdown ─────────────────────────────────────────────
    pdf.section_title("Damage Type Breakdown", BLUE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 230, 240)
    for cls, count in stats.get("by_class", {}).items():
        pct = count / max(stats["total"], 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(40, 7, cls.capitalize())
        pdf.cell(12, 7, str(count), align="R")
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(*BLUE)
        pdf.cell(60, 7, f"  {bar}")
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, f"{pct:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Model metrics (if available) ───────────────────────────────────
    if model_metrics:
        pdf.section_title("Model Performance Metrics", BLUE)
        rows = [
            ("mAP50",      f"{model_metrics.get('mAP50',     0):.4f}"),
            ("Precision",  f"{model_metrics.get('precision', 0):.4f}"),
            ("Recall",     f"{model_metrics.get('recall',    0):.4f}"),
            ("F1 Score",   f"{model_metrics.get('f1',        0):.4f}"),
        ]
        for label, val in rows:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(60, 7, label)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    # ── Detailed detection table (up to 50 rows) ───────────────────────
    pdf.add_page()
    pdf.section_title("Detailed Detection Log (first 50)", DARK)

    # Table header
    col_w  = [10, 22, 18, 25, 50, 45]
    header = ["#", "Class", "Severity", "Confidence", "GPS Coordinates", "Timestamp"]
    pdf.set_fill_color(30, 30, 46)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 8)
    for i, (h_txt, w) in enumerate(zip(header, col_w)):
        pdf.cell(w, 7, h_txt, border=1, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    det_list = list(detections)
    for i, det in enumerate(det_list[:50]):  # type: ignore
        sev  = det.get("severity", "LOW")
        gps  = det.get("gps", {})
        fill = i % 2 == 0
        color_map = {"HIGH": (255, 220, 220), "MEDIUM": (255, 245, 200), "LOW": (220, 255, 220)}
        pdf.set_fill_color(*color_map.get(sev, (245, 245, 245)))
        pdf.set_text_color(30, 30, 30)

        cells = [
            str(i + 1),
            det.get("class", "?"),
            sev,
            f"{det.get('confidence', 0):.2%}",
            f"{gps.get('lat', 0):.6f}, {gps.get('lon', 0):.6f}",
            det.get("timestamp", "")[:19],
        ]
        for val, w in zip(cells, col_w):
            pdf.cell(w, 6, val, border=1, fill=True)
        pdf.ln()

    if len(detections) > 50:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 8, f"  ... and {len(detections) - 50} more detections (see CSV report)")
        pdf.ln()

    # ── Recommendations ────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Maintenance Recommendations", RED if stats["high"] > 0 else GREEN)
    recs = _generate_recommendations(stats)
    pdf.set_font("Helvetica", "", 10)
    for i, rec in enumerate(recs, 1):
        pdf.multi_cell(0, 7, f"{i}. {rec}")
        pdf.ln(1)

    # ── Footer signature ───────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(0, 6,
        f"This report was automatically generated by the AI Road Quality Monitoring System "
        f"on {date_str} at {time_str}. All damage detections are AI-assisted and should be "
        f"verified by a qualified road engineer before maintenance work is commissioned."
    )

    # ── Output ─────────────────────────────────────────────────────────
    if output_path:
        pdf.output(str(output_path))
        print(f"✅ PDF report saved: {output_path}")

    return bytes(pdf.output())


def generate_csv_report(
    detections:  list,
    output_path: Optional[Path] = None,
) -> bytes:
    """
    Generate a CSV report of all detections.

    Returns:
        CSV content as bytes (for Streamlit download button)
    """
    fieldnames = [
        "id", "timestamp", "class", "severity", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "gps_lat", "gps_lon", "gps_alt", "gps_speed",
        "repair_priority",
    ]
    priority_map = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    for i, det in enumerate(detections):
        gps  = det.get("gps", {})
        bbox = det.get("bbox", [0, 0, 0, 0])
        sev  = det.get("severity", "LOW")

        writer.writerow({
            "id":              i + 1,
            "timestamp":       det.get("timestamp", ""),
            "class":           det.get("class", ""),
            "severity":        sev,
            "confidence":      round(det.get("confidence", 0.0), 4),
            "bbox_x1":         bbox[0] if len(bbox) > 0 else 0,
            "bbox_y1":         bbox[1] if len(bbox) > 1 else 0,
            "bbox_x2":         bbox[2] if len(bbox) > 2 else 0,
            "bbox_y2":         bbox[3] if len(bbox) > 3 else 0,
            "gps_lat":         gps.get("lat", 0.0),
            "gps_lon":         gps.get("lon", 0.0),
            "gps_alt":         gps.get("alt", 0.0),
            "gps_speed":       gps.get("speed", 0.0),
            "repair_priority": priority_map.get(sev, 3),
        })

    csv_bytes = buf.getvalue().encode("utf-8")

    if output_path is not None:
        p_str = str(output_path)
        with open(p_str, "wb") as f:
            f.write(csv_bytes)
        print(f"✅ CSV report saved: {output_path}")

    return csv_bytes


# ── Helper functions ────────────────────────────────────────────────────
def _compute_stats(detections: list) -> dict:
    total  = len(detections)
    high   = sum(1 for d in detections if d.get("severity") == "HIGH")
    medium = sum(1 for d in detections if d.get("severity") == "MEDIUM")
    low    = sum(1 for d in detections if d.get("severity") == "LOW")
    by_cls = {}
    for d in detections:
        k = d.get("class", "unknown")
        by_cls[k] = by_cls.get(k, 0) + 1
    return {"total": total, "high": high, "medium": medium, "low": low, "by_class": by_cls}


def _overall_condition(stats: dict) -> str:
    total = max(stats["total"], 1)
    if stats["high"] / total > 0.3:
        return "POOR — Urgent attention required"
    elif stats["high"] / total > 0.1 or stats["medium"] / total > 0.4:
        return "FAIR — Maintenance required"
    elif stats["medium"] / total > 0.2:
        return "MODERATE — Schedule maintenance"
    else:
        return "GOOD — Routine monitoring recommended"


def _generate_recommendations(stats: dict) -> list:
    recs = []
    if stats["high"] > 0:
        recs.append(
            f"URGENT: {stats['high']} high-severity defects require immediate response. "
            "Deploy patching crew within 48 hours to prevent vehicle damage and accidents."
        )
    if stats.get("by_class", {}).get("pothole", 0) > 5:
        recs.append(
            "Multiple potholes detected. Consider cold-mix asphalt patching for temporary "
            "fixes and schedule full resurfacing during the dry season."
        )
    if stats.get("by_class", {}).get("crack", 0) > 3:
        recs.append(
            "Significant cracking detected. Apply polymer-modified bitumen crack sealant "
            "to prevent water infiltration and structural deterioration."
        )
    if stats["medium"] > 0:
        recs.append(
            f"{stats['medium']} medium-severity locations should be addressed within 2–4 weeks "
            "as part of routine maintenance scheduling."
        )
    if stats["low"] > 0:
        recs.append(
            f"{stats['low']} low-severity defects should be monitored and included in the "
            "next planned maintenance cycle."
        )
    recs.append(
        "Consider installing road condition monitoring sensors (IoT) at identified "
        "high-damage zones for continuous monitoring between surveys."
    )
    recs.append(
        "Update the road asset management database with GPS-tagged damage records "
        "from this survey for future planning and resource allocation."
    )
    return recs
