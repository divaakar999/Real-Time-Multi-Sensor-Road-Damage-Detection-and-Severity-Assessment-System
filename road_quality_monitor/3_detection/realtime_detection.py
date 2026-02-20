"""
=======================================================================
STEP 3A - REAL-TIME DETECTION: Live Webcam / Dashcam Inference
=======================================================================
Runs YOLOv8 on:
  - Webcam      (source=0)
  - Video file  (source=video.mp4)
  - RTSP stream (source=rtsp://...)
  - Image       (source=image.jpg)

Features:
  • Real-time bounding box overlay
  • Severity classification (Low/Medium/High)
  • FPS counter
  • GPS coordinate tagging per detection
  • Auto-save detections with metadata

Usage:
  python realtime_detection.py --source 0
  python realtime_detection.py --source dashcam.mp4
  python realtime_detection.py --source rtsp://192.168.1.100/stream
=======================================================================
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

from ultralytics import YOLO
from severity_classifier import SeverityClassifier
from gps_tagger import GPSTagger

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
WEIGHTS_PATH   = PROJECT_ROOT / "weights" / "best.pt"
DETECTION_LOG  = PROJECT_ROOT / "detections.json"
SAVED_FRAMES   = PROJECT_ROOT / "saved_detections"
SAVED_FRAMES.mkdir(exist_ok=True)

# ── Class names & colours ──────────────────────────────────────────────
CLASS_NAMES    = ["pothole", "crack", "wear"]
CLASS_COLORS   = {
    "pothole": (0,   80, 255),   # Red-Orange (BGR)
    "crack":   (0, 200, 255),   # Yellow
    "wear":    (0, 165, 255),   # Orange
}
SEVERITY_COLORS = {
    "HIGH":   (0,  0,  220),    # Red
    "MEDIUM": (0, 165, 255),    # Orange
    "LOW":    (50, 200,  50),   # Green
}


class RoadDamageDetector:
    """
    Real-time road damage detector using YOLOv8.

    Attributes:
        model:            Loaded YOLOv8 model
        severity:         SeverityClassifier instance
        gps:              GPSTagger instance
        conf_threshold:   Minimum confidence to show detection (0–1)
        detections_log:   In-memory list of all detections this session
    """

    def __init__(
        self,
        weights_path : Path  = WEIGHTS_PATH,
        conf         : float = 0.40,
        iou          : float = 0.45,
        save_interval: int   = 30,    # Save frame every N detections
    ):
        print(f"\n🔃 Loading YOLOv8 model: {weights_path}")

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weights_path}\n"
                "Train first: python 2_model/train_yolov8.py"
            )

        self.model          = YOLO(str(weights_path))
        self.conf           = conf
        self.iou            = iou
        self.save_interval  = save_interval
        self.severity_clf   = SeverityClassifier()
        self.gps_tagger     = GPSTagger()
        self.detections_log = []
        self.frame_count    = 0
        self.detection_count= 0

        # FPS tracking (rolling average over last 30 frames)
        self.fps_deque      = deque(maxlen=30)
        self._last_tick     = datetime.now()

        print(f"✅ Model loaded | conf={conf} | iou={iou}")
        print(f"   Classes: {CLASS_NAMES}\n")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Run detection on a single frame and draw results.

        Args:
            frame: BGR image (from cv2.VideoCapture)

        Returns:
            annotated_frame: Frame with bounding boxes drawn
            detections:      List of detection dicts for this frame
        """
        h, w   = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf    = self.conf,
            iou     = self.iou,
            verbose = False,
        )

        detections       = []
        annotated_frame  = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # ── Extract detection info ─────────────────────────────
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score      = float(box.conf[0])
                class_id        = int(box.cls[0])
                
                # Use model.names instead of hardcoded CLASS_NAMES
                real_class      = self.model.names.get(class_id, f"class_{class_id}")
                
                ROAD_DAMAGE_CLASSES = {
                    "pothole", "crack", "wear",
                    "alligator crack", "longitudinal crack",
                    "transverse crack", "depression", "raveling",
                    "rutting", "bleeding", "damage",
                }
                
                is_road_damage = real_class.lower() in ROAD_DAMAGE_CLASSES

                if is_road_damage:
                    class_name = real_class
                    # ── Classify severity ──────────────────────────────────
                    severity = self.severity_clf.classify(
                        bbox        = (x1, y1, x2, y2),
                        frame_shape = (h, w),
                        class_name  = class_name,
                        confidence  = conf_score,
                    )

                    # ── Get GPS ───────────────────────────────────────────
                    gps = self.gps_tagger.get_current()

                    detection = {
                        "frame":      self.frame_count,
                        "class":      class_name,
                        "confidence": round(conf_score, 3),
                        "severity":   severity,
                        "bbox":       [x1, y1, x2, y2],
                        "gps":        gps,
                        "timestamp":  datetime.now().isoformat(),
                    }
                    detections.append(detection)
                    self.detections_log.append(detection)
                    self.detection_count += 1

                    # ── Draw bounding box ──────────────────────────────────
                    color  = SEVERITY_COLORS.get(severity, (255, 255, 255))
                    thick  = 3 if severity == "HIGH" else 2

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thick)

                    # Label background
                    label       = f"{class_name} [{severity}] {conf_score:.2f}"
                else:
                    # Non-road class
                    color = (180, 180, 180) # Grey
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{real_class} {conf_score:.2f} [not road]"

                label_size  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
                label_bg_y2 = max(y1 - 4, label_size[1] + 4)
                cv2.rectangle(annotated_frame,
                              (x1, label_bg_y2 - label_size[1] - 4),
                              (x1 + label_size[0] + 4, label_bg_y2),
                              color, -1)
                cv2.putText(annotated_frame, label,
                            (x1 + 2, label_bg_y2 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # ── Auto-save frames with detections ──────────────────────────
        if detections and (self.detection_count % self.save_interval == 0):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = SAVED_FRAMES / f"detection_{ts}.jpg"
            cv2.imwrite(str(path), annotated_frame)

        self.frame_count += 1
        return annotated_frame, detections

    def draw_hud(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw Heads-Up Display overlay with FPS, GPS, detection counts."""
        h, w      = frame.shape[:2]
        overlay   = frame.copy()
        gps       = self.gps_tagger.get_current()

        # ── Semi-transparent HUD panel ────────────────────────────────
        cv2.rectangle(overlay, (0, 0), (350, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        font    = cv2.FONT_HERSHEY_SIMPLEX
        ts      = datetime.now().strftime("%H:%M:%S")

        hud_lines = [
            (f"FPS: {fps:5.1f}",                  (0, 255, 0)),
            (f"Total Detections: {self.detection_count}", (0, 200, 255)),
            (f"GPS: {gps['lat']:.5f}, {gps['lon']:.5f}", (255, 200, 0)),
            (f"Time: {ts}",                        (200, 200, 200)),
        ]

        for i, (text, color) in enumerate(hud_lines):
            cv2.putText(frame, text, (8, 22 + i * 22), font, 0.55, color, 1)

        # ── Severity legend ───────────────────────────────────────────
        legend = [("HIGH",   SEVERITY_COLORS["HIGH"]),
                  ("MEDIUM", SEVERITY_COLORS["MEDIUM"]),
                  ("LOW",    SEVERITY_COLORS["LOW"])]
        for i, (sev, col) in enumerate(legend):
            x = w - 120
            y = 20 + i * 25
            cv2.rectangle(frame, (x, y - 14), (x + 16, y), col, -1)
            cv2.putText(frame, sev, (x + 22, y - 2), font, 0.5, (255, 255, 255), 1)

        return frame

    def save_detections_log(self):
        """Save all detections to JSON for the dashboard to read."""
        with open(DETECTION_LOG, "w") as f:
            json.dump(self.detections_log, f, indent=2)
        print(f"💾 Detections saved: {DETECTION_LOG}")
        print(f"   Total detections: {self.detection_count}")


def run_detection(source, show: bool = True, record: bool = False):
    """
    Main detection loop.

    Args:
        source: Camera index (int), video path (str), or RTSP URL
        show:   Display real-time window
        record: Save output video
    """
    detector = RoadDamageDetector()

    # ── Open video source ──────────────────────────────────────────────
    cap_src = int(source) if str(source).isdigit() else source
    cap     = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Source: {source}  |  Resolution: {w}×{h}  |  FPS: {fps_src:.0f}")

    # ── Optional: video recording ──────────────────────────────────────
    writer = None
    if record:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_vid = PROJECT_ROOT / f"recording_{ts}.mp4"
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(str(out_vid), fourcc, fps_src, (w, h))
        print(f"🎥 Recording to: {out_vid}")

    import time
    prev_time = time.perf_counter()

    print("\n▶  Detection running. Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Stream ended.")
                break

            # ── Run detection ──────────────────────────────────────────
            annotated, detections = detector.process_frame(frame)

            # ── Compute FPS ────────────────────────────────────────────
            now      = time.perf_counter()
            fps      = 1.0 / (now - prev_time + 1e-10)
            prev_time= now
            detector.fps_deque.append(fps)
            avg_fps  = np.mean(detector.fps_deque)

            # ── Draw HUD ───────────────────────────────────────────────
            annotated = detector.draw_hud(annotated, avg_fps)

            # Print detections
            for det in detections:
                print(f"  🔴 [{det['severity']:6s}] {det['class']:8s} "
                      f"conf={det['confidence']:.2f}  "
                      f"GPS({det['gps']['lat']:.5f}, {det['gps']['lon']:.5f})")

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("Road Quality Monitor  [q=quit]", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n⏹  Stopped by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        detector.save_detections_log()
        print(f"\n✅ Detection session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time road damage detection")
    parser.add_argument("--source",  default="0",          help="Camera/video source")
    parser.add_argument("--noshow",  action="store_true",  help="Disable video window")
    parser.add_argument("--record",  action="store_true",  help="Save output video")
    args = parser.parse_args()

    run_detection(
        source = args.source,
        show   = not args.noshow,
        record = args.record,
    )
