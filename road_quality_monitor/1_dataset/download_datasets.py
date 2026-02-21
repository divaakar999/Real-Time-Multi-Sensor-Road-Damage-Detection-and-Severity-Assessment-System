"""
=======================================================================
STEP 1A - DATASET: Download Open-Source Road Damage Datasets
=======================================================================

Best Open-Source Datasets for Road Damage Detection:

1. RDD2022 (Road Damage Dataset 2022) ← BEST CHOICE
   - Source: IEEE DataPort / Paperswithcode
   - 47,420 images from Japan, India, Czech Republic, Norway, USA
   - Classes: D00 (longitudinal crack), D10 (transverse crack),
              D20 (alligator crack), D40 (pothole)
   - URL: https://github.com/sekilab/RoadDamageDetector

2. Pothole-600 Dataset
   - 600 labelled pothole images
   - URL: https://universe.roboflow.com/university-sbxzs/pothole-detection-a73yl

3. Road Surface Condition Dataset (Roboflow Universe)
   - URL: https://universe.roboflow.com

4. CRDDC2022 (Crowdsensing Road Damage Dataset Challenge)
   - Multi-country, multi-damage type

This script downloads datasets and organises them for YOLOv8.
=======================================================================
"""

import os
import zipfile
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"


def create_directory_structure():
    """Create the data folder structure expected by YOLOv8."""
    dirs = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR / "images" / "train",
        PROCESSED_DIR / "images" / "val",
        PROCESSED_DIR / "images" / "test",
        PROCESSED_DIR / "labels" / "train",
        PROCESSED_DIR / "labels" / "val",
        PROCESSED_DIR / "labels" / "test",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created.")


def download_file(url: str, dest: Path, description: str = "Downloading") -> Path:
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total    = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f, tqdm(
        desc=description, total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def download_via_roboflow():
    """
    Download dataset directly from Roboflow.
    You need a FREE Roboflow account and API key.

    Steps to get your API key:
      1. Go to https://roboflow.com and create a FREE account
      2. Navigate to Settings → API Keys
      3. Copy your Private API Key
      4. Paste it below or set the environment variable ROBOFLOW_API_KEY
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow first: pip install roboflow")
        return False

    api_key = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")

    if api_key == "YOUR_API_KEY_HERE":
        print(
            "\n⚠️  ROBOFLOW API KEY REQUIRED\n"
            "   1. Sign up free at https://roboflow.com\n"
            "   2. Get your API key from Settings → API Keys\n"
            "   3. Run:  set ROBOFLOW_API_KEY=your_key_here\n"
            "   4. Re-run this script\n"
        )
        return False

    rf      = Roboflow(api_key=api_key)

    # ── Option A: Pothole Detection dataset (most popular) ──
    project = rf.workspace("university-sbxzs").project("pothole-detection-a73yl")
    version = project.version(1)
    dataset = version.download("yolov8", location=str(PROCESSED_DIR))

    print(f"✅ Dataset downloaded to: {PROCESSED_DIR}")
    return True


def download_rdd2022_sample():
    """
    Download a curated sample from the RDD2022 dataset.
    Full dataset available at: https://github.com/sekilab/RoadDamageDetector
    """
    print("\n📦 Downloading RDD2022 sample data...")
    print("   Full dataset: https://github.com/sekilab/RoadDamageDetector")
    print("   Paper: https://arxiv.org/abs/2209.08538\n")

    # Instructions for manual download:
    instructions = """
    MANUAL DOWNLOAD STEPS FOR RDD2022:
    ===================================
    1. Visit: https://github.com/sekilab/RoadDamageDetector
    2. Follow the data download instructions
    3. Extract to: data/raw/rdd2022/
    4. Run the convert_rdd_to_yolo() function below

    Alternative - Direct Google Drive links (from paper repo):
    - Japan:       https://github.com/sekilab/RoadDamageDetector
    - India:       (same repo)
    - Czech:       (same repo)
    """
    print(instructions)


def convert_rdd_to_yolo(rdd_root: Path = RAW_DIR / "rdd2022"):
    """
    Convert RDD2022 Pascal VOC XML annotations → YOLOv8 TXT format.

    RDD2022 class mapping:
      D00 → 0 (Longitudinal Crack)
      D10 → 1 (Transverse Crack)
      D20 → 2 (Alligator Crack)
      D40 → 3 (Pothole)
    """
    import xml.etree.ElementTree as ET

    CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}

    xml_files = list(rdd_root.rglob("*.xml"))
    if not xml_files:
        print(f"⚠️  No XML files found in {rdd_root}")
        return

    print(f"\n🔄 Converting {len(xml_files)} XML annotations to YOLO format...")

    for xml_path in tqdm(xml_files):
        tree   = ET.parse(xml_path)
        root   = tree.getroot()

        # Get image dimensions
        size   = root.find("size")
        width  = int(size.find("width").text)
        height = int(size.find("height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[class_name]
            bbox     = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Convert to YOLO format: class cx cy w h (all normalized 0-1)
            cx = (xmin + xmax) / 2 / width
            cy = (ymin + ymax) / 2 / height
            w  = (xmax - xmin) / width
            h  = (ymax - ymin) / height

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Save label file
        label_path = PROCESSED_DIR / "labels" / "train" / (xml_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"✅ Conversion complete. Labels saved to: {PROCESSED_DIR / 'labels'}")


def split_dataset(split_ratio: tuple = (0.7, 0.2, 0.1)):
    """
    Automatically split dataset into train / val / test sets.

    Args:
        split_ratio: (train, val, test) ratios (must sum to 1.0)
    """
    import random
    from shutil import copy2

    all_images = list((PROCESSED_DIR / "images" / "train").glob("*.jpg"))
    all_images += list((PROCESSED_DIR / "images" / "train").glob("*.png"))

    if not all_images:
        print("⚠️  No images found for splitting. Run download first.")
        return

    random.shuffle(all_images)
    n      = len(all_images)
    n_val  = int(n * split_ratio[1])
    n_test = int(n * split_ratio[2])

    val_imgs  = all_images[:n_val]
    test_imgs = all_images[n_val:n_val + n_test]

    for img_path in val_imgs:
        label_path = PROCESSED_DIR / "labels" / "train" / (img_path.stem + ".txt")
        copy2(img_path, PROCESSED_DIR / "images" / "val")
        if label_path.exists():
            copy2(label_path, PROCESSED_DIR / "labels" / "val")

    for img_path in test_imgs:
        label_path = PROCESSED_DIR / "labels" / "train" / (img_path.stem + ".txt")
        copy2(img_path, PROCESSED_DIR / "images" / "test")
        if label_path.exists():
            copy2(label_path, PROCESSED_DIR / "labels" / "test")

    print(f"✅ Dataset split: {n - n_val - n_test} train | {n_val} val | {n_test} test")


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Road Quality Monitor - Dataset Setup")
    print("=" * 60)

    create_directory_structure()

    print("\nChoose dataset source:")
    print("  1. Roboflow (recommended - requires free API key)")
    print("  2. RDD2022  (manual download instructions)")
    print("  3. Skip (use your own images)")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        success = download_via_roboflow()
        if success:
            split_dataset()
    elif choice == "2":
        download_rdd2022_sample()
        print("\nAfter downloading, run: python download_datasets.py --convert")
    else:
        print("\n📌 Place your images in: data/processed/images/train/")
        print("📌 Place your labels in: data/processed/labels/train/")
        print("📌 Then run:  python download_datasets.py --split")
