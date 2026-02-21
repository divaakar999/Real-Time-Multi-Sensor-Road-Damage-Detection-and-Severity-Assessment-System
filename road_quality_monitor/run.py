"""
=======================================================================
ONE-CLICK LAUNCHER
=======================================================================
Run everything from a single entry point:
    python run.py --setup         Install requirements
    python run.py --dashboard     Launch Streamlit dashboard
    python run.py --detect        Start live detection
    python run.py --train         Train YOLOv8
    python run.py --evaluate      Evaluate trained model
    python run.py --compare       Generate comparison charts
    python run.py --test          Test all modules (no model needed)
=======================================================================
"""

import sys
import io
import os
import subprocess
import argparse
from pathlib import Path

# Force UTF-8 output on Windows to avoid emoji/unicode errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent


def run(cmd: str, cwd: Path = ROOT):
    """Run a shell command and stream output."""
    print(f"\n▶  {cmd}\n{'─'*60}")
    result = subprocess.run(cmd, shell=True, cwd=str(cwd))
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
    return result.returncode


def check_requirements():
    """Check which required packages are installed."""
    deps  = [
        ("ultralytics",      "ultralytics"),
        ("cv2",              "opencv-python"),
        ("streamlit",        "streamlit"),
        ("folium",           "folium"),
        ("streamlit_folium", "streamlit-folium"),
        ("fpdf",             "fpdf2"),
        ("plotly",           "plotly"),
        ("pandas",           "pandas"),
        ("numpy",            "numpy"),
        ("sklearn",          "scikit-learn"),
        ("roboflow",         "roboflow"),
    ]
    print("\n📦 Checking installed packages:")
    missing = []
    for import_name, pip_name in deps:
        try:
            __import__(import_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} (missing)")
            missing.append(pip_name)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    print("\n✅ All packages installed!")
    return True


def test_modules():
    """Quick smoke test of all modules without requiring model or camera."""
    print("\n🧪 Running module tests...\n")

    # Test severity classifier
    print("─" * 40)
    print("Test 1: Severity Classifier")
    sys.path.insert(0, str(ROOT / "3_detection"))
    try:
        from severity_classifier import SeverityClassifier
        clf = SeverityClassifier()
        result = clf.classify((100, 100, 400, 300), (720, 1280), "pothole", 0.8)
        print(f"  ✅ SeverityClassifier works | Result: {result}")
    except Exception as e:
        print(f"  ❌ SeverityClassifier failed: {e}")

    # Test GPS tagger
    print("─" * 40)
    print("Test 2: GPS Tagger")
    try:
        from gps_tagger import GPSTagger
        gps = GPSTagger()
        pt  = gps.get_current()
        print(f"  ✅ GPSTagger works | lat={pt['lat']}, lon={pt['lon']}")
        dist = GPSTagger.haversine_distance(
            {"lat": 12.971, "lon": 77.594},
            {"lat": 12.972, "lon": 77.595},
        )
        print(f"  ✅ Haversine distance: {dist} metres")
    except Exception as e:
        print(f"  ❌ GPSTagger failed: {e}")

    # Test map component
    print("─" * 40)
    print("Test 3: Map Component")
    sys.path.insert(0, str(ROOT / "4_dashboard"))
    try:
        from map_component import build_damage_map, get_map_statistics
        sample_dets = [
            {"class":"pothole","confidence":0.8,"severity":"HIGH",
             "gps":{"lat":12.971,"lon":77.594},"timestamp":"2026-01-01T00:00:00"},
        ]
        fmap = build_damage_map(sample_dets)
        stats = get_map_statistics(sample_dets)
        print(f"  ✅ Map component works | stats={stats}")
    except Exception as e:
        print(f"  ❌ Map component failed: {e}")

    # Test report generator
    print("─" * 40)
    print("Test 4: Report Generator (CSV only, no fpdf2 required)")
    try:
        from report_generator import generate_csv_report
        csv_bytes = generate_csv_report(sample_dets)
        print(f"  ✅ CSV report works | {len(csv_bytes)} bytes")
    except Exception as e:
        print(f"  ❌ Report generator failed: {e}")

    print("\n✅ Module tests complete!")


def main():
    parser = argparse.ArgumentParser(
        description = "AI Road Quality Monitoring System — Launcher",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --setup        # Install all dependencies
  python run.py --test         # Test all modules (no model needed)
  python run.py --dashboard    # Launch Streamlit dashboard
  python run.py --detect       # Start live webcam detection
  python run.py --train        # Train YOLOv8 model
  python run.py --evaluate     # Evaluate model performance
  python run.py --compare      # Generate comparison charts
        """
    )

    parser.add_argument("--setup",     action="store_true", help="Install requirements")
    parser.add_argument("--test",      action="store_true", help="Test all modules")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--detect",    action="store_true", help="Start live detection")
    parser.add_argument("--train",     action="store_true", help="Train YOLOv8")
    parser.add_argument("--evaluate",  action="store_true", help="Evaluate model")
    parser.add_argument("--compare",   action="store_true", help="Comparison charts")
    parser.add_argument("--source",    default="0",          help="Video source for detection")

    args = parser.parse_args()

    print("=" * 60)
    print("  🛣️  AI Road Quality Monitoring System")
    print("=" * 60)

    if args.setup:
        print("\n📦 Installing requirements...")
        run(f"{sys.executable} -m pip install -r requirements.txt")
        check_requirements()

    elif args.test:
        check_requirements()
        test_modules()

    elif args.dashboard:
        print("\n🌐 Launching Streamlit dashboard at http://localhost:8501")
        run(f"streamlit run 4_dashboard/app.py --server.port 8501 --theme.base dark")

    elif args.detect:
        print(f"\n🎥 Starting live detection (source={args.source})")
        run(f"{sys.executable} 3_detection/realtime_detection.py --source {args.source}")

    elif args.train:
        print("\n🚀 Starting YOLOv8 training...")
        run(f"{sys.executable} 2_model/train_yolov8.py")

    elif args.evaluate:
        print("\n📊 Evaluating model...")
        run(f"{sys.executable} 2_model/evaluate_model.py")

    elif args.compare:
        print("\n📊 Generating comparison charts...")
        run(f"{sys.executable} 5_evaluation/benchmark.py --compare")

    else:
        # No args — show menu
        print("""
  What would you like to do?

  1. Install requirements      →  python run.py --setup
  2. Test all modules          →  python run.py --test
  3. Launch dashboard          →  python run.py --dashboard
  4. Live webcam detection     →  python run.py --detect
  5. Train model               →  python run.py --train
  6. Evaluate model            →  python run.py --evaluate
  7. Compare with baselines    →  python run.py --compare
        """)


if __name__ == "__main__":
    main()
