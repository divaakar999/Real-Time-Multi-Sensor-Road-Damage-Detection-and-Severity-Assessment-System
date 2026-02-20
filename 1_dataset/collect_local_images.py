"""
=======================================================================
STEP 1B - DATASET: Collect Your Own Local Road Images
=======================================================================
Use this script to:
  - Capture road images from a webcam or dashcam
  - Automatically save frames at timed intervals
  - Use geopy + GPS device for coordinate tagging

HOW TO USE:
  python collect_local_images.py --source 0           # webcam
  python collect_local_images.py --source video.mp4   # dashcam video
  python collect_local_images.py --source rtsp://...  # IP camera/RTSP
=======================================================================
"""

import cv2
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Optional: enable for real GPS hardware (requires gpsd daemon)
USE_REAL_GPS = False


def get_gps_coordinates():
    """
    Get real GPS coordinates from a connected GPS device (via gpsd).
    Falls back to simulated coordinates for testing.

    For real GPS:
      1. Install gpsd:  sudo apt install gpsd gpsd-clients
      2. Connect GPS USB dongle
      3. Run:  gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock
      4. Set USE_REAL_GPS = True
    """
    if USE_REAL_GPS:
        try:
            import gpsd
            gpsd.connect()
            packet = gpsd.get_current()
            return {
                "lat":  round(packet.lat, 7),
                "lon":  round(packet.lon, 7),
                "alt":  round(packet.alt, 2),
                "speed": round(packet.speed(), 2),
                "time": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            print(f"⚠️  GPS error: {e}. Using simulated coordinates.")

    # ── Simulated GPS (for testing without hardware) ─────────────────
    # Replace with your actual area's coordinates
    import math
    # Simulate slight movement along a road
    t   = time.time()
    lat = 12.9716 + (t % 1000) * 0.00001   # Bangalore base coords
    lon = 77.5946 + (t % 1000) * 0.00001
    return {
        "lat":  round(lat, 7),
        "lon":  round(lon, 7),
        "alt":  0.0,
        "speed": 30.0,
        "time": datetime.utcnow().isoformat(),
        "simulated": True,
    }


def collect_images(
    source,
    output_dir: Path,
    interval_seconds: float = 0.5,
    max_images: int = 1000,
    blur_threshold: float = 100.0,
):
    """
    Capture frames from a video source and save them with GPS metadata.

    Args:
        source:           Camera index (0,1,2) or video file path or RTSP URL
        output_dir:       Where to save captured images
        interval_seconds: Minimum time between captured frames (seconds)
        max_images:       Stop after this many images
        blur_threshold:   Reject blurry frames below this Laplacian variance
    """
    output_dir   = Path(output_dir)
    images_dir   = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Open video source
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"\n📸 Starting collection from source: {source}")
    print(f"   Output:   {output_dir}")
    print(f"   Interval: every {interval_seconds}s")
    print(f"   Max:      {max_images} images")
    print("   Press 'q' to quit, 's' to force save current frame\n")

    saved_count  = 0
    last_save    = 0
    all_metadata = []

    while saved_count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Stream ended or no frame received.")
            break

        # ── Blur detection (reject unsharp frames) ──────────────────
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Overlay info on preview
        preview = frame.copy()
        status  = "SHARP" if variance >= blur_threshold else "BLURRY"
        color   = (0, 255, 0) if variance >= blur_threshold else (0, 0, 255)
        cv2.putText(
            preview, f"Blur: {variance:.0f} ({status})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        cv2.putText(
            preview, f"Saved: {saved_count}/{max_images}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        cv2.imshow("Road Image Collector  [q=quit | s=save]", preview)

        key     = cv2.waitKey(1) & 0xFF
        now     = time.time()
        force   = (key == ord("s"))

        # ── Save logic ───────────────────────────────────────────────
        should_save = (
            (now - last_save >= interval_seconds) and
            (variance >= blur_threshold or force)
        )

        if should_save or force:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_name   = f"road_{timestamp}.jpg"
            img_path   = images_dir / img_name
            gps        = get_gps_coordinates()

            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            meta = {
                "filename":  img_name,
                "gps":       gps,
                "blur_score": round(float(variance), 2),
                "timestamp": timestamp,
            }
            all_metadata.append(meta)

            # Save per-image metadata
            with open(metadata_dir / f"{img_name}.json", "w") as mf:
                json.dump(meta, mf, indent=2)

            saved_count += 1
            last_save    = now
            print(f"  ✅ Saved {img_name} | GPS: {gps['lat']}, {gps['lon']} | Blur: {variance:.0f}")

        if key == ord("q"):
            break

    # Save combined metadata
    master_meta_path = output_dir / "metadata.json"
    with open(master_meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Collection complete!")
    print(f"   Total images saved: {saved_count}")
    print(f"   Metadata file:      {master_meta_path}")
    print(f"\n📌 Next step: Annotate images using Roboflow")
    print(f"   1. Upload to https://roboflow.com")
    print(f"   2. Use the annotation tool to draw bounding boxes")
    print(f"   3. Export in YOLOv8 format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect road images for dataset")
    parser.add_argument("--source",   default="0",    help="Camera index, video file, or RTSP URL")
    parser.add_argument("--output",   default="data/raw/local_collection", help="Output directory")
    parser.add_argument("--interval", default=0.5,   type=float, help="Seconds between captures")
    parser.add_argument("--max",      default=1000,  type=int,   help="Maximum images to capture")
    parser.add_argument("--blur",     default=100.0, type=float, help="Blur threshold (higher=sharper)")
    args = parser.parse_args()

    collect_images(
        source           = args.source,
        output_dir       = Path(args.output),
        interval_seconds = args.interval,
        max_images       = args.max,
        blur_threshold   = args.blur,
    )
