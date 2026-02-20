"""
setup_demo_model.py
────────────────────────────────────────────────────────────────────────
Downloads the base YOLOv8n pretrained model and copies it to
weights/best.pt so the dashboard "Live Detection" mode works
immediately — even before you train your custom road damage model.

The base yolov8n.pt can detect general objects (not road-specific),
but is useful for smoke-testing the pipeline end-to-end.

Run:
    python setup_demo_model.py

After real training is done, replace weights/best.pt with your
trained model from:
    runs/detect/road_damage_XXXXXX/weights/best.pt
────────────────────────────────────────────────────────────────────────
"""

import sys
import shutil
from pathlib import Path

ROOT         = Path(__file__).parent
WEIGHTS_DIR  = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
TARGET       = WEIGHTS_DIR / "best.pt"

def main():
    print("=" * 60)
    print("  Road Quality Monitor — Demo Model Setup")
    print("=" * 60)

    if TARGET.exists() and TARGET.stat().st_size > 1000:
        print(f"\n✔  {TARGET} already exists ({TARGET.stat().st_size / 1e6:.1f} MB)")
        print("   Delete it if you want to re-download.")
        return

    print("\n📦 Downloading base YOLOv8n pretrained weights (~6 MB)...")
    print("   (This is the COCO-pretrained model, not road-specific)")
    print("   It detects 80 general object classes as a demo.\n")

    try:
        from ultralytics import YOLO

        # Download the nano model (fastest, smallest)
        model = YOLO("yolov8n.pt")           # auto-downloads from ultralytics CDN
        src   = Path(model.ckpt_path)

        shutil.copy(src, TARGET)
        print(f"\n✅ Copied {src.name} → {TARGET}")
        print(f"   Size: {TARGET.stat().st_size / 1e6:.1f} MB")

    except Exception as e:
        print(f"❌ Failed: {e}")
        print("   Try manually: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
        sys.exit(1)

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DONE! Now you can:
  
  1. Launch dashboard and use "Live Detection" mode:
       python -X utf8 -m streamlit run 4_dashboard/app.py

  2. The demo model detects general objects (person, car, etc.)
     For road damage detection, train your own model:

       python -X utf8 2_model/train_yolov8.py
       # OR open colab_training.ipynb in Google Colab

  3. After training, replace weights/best.pt:
       copy runs\\detect\\road_damage_XXXX\\weights\\best.pt weights\\best.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
