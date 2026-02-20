"""
=======================================================================
STEP 2A - MODEL: Train YOLOv8 on Road Damage Dataset
=======================================================================

YOLOv8 Model Size Options (choose based on your hardware):
  yolov8n  ← Nano    (fastest, lowest accuracy, good for embedded)
  yolov8s  ← Small   (good balance for edge devices)
  yolov8m  ← Medium  (recommended ✅ for final year project)
  yolov8l  ← Large   (better accuracy, needs GPU)
  yolov8x  ← Extra   (best accuracy, slowest)

Training on CPU: yolov8n or yolov8s only (very slow)
Training on GPU (NVIDIA): Any model, recommended yolov8m

TRAINING CONCEPTS EXPLAINED:
─────────────────────────────
• Epochs:      How many times the model sees the entire dataset
               50–100 epochs is typical for road damage
• Batch Size:  Images processed together (larger = faster but more VRAM)
• Image Size:  640×640 is YOLOv8 standard input
• Optimizer:   SGD or AdamW (auto-selected by YOLOv8)
• Learning Rate: How fast weights update. YOLOv8 auto-schedules this.

EVALUATION METRICS:
───────────────────
• mAP50:      Mean Average Precision at 50% IoU threshold
               > 0.80 = Good | > 0.90 = Excellent
• Precision:  Of all detected boxes, how many are correct?
               TP / (TP + FP)
• Recall:     Of all actual damage, how much did we find?
               TP / (TP + FN)
• F1 Score:   Harmonic mean of Precision & Recall
• IoU:        Intersection over Union — measures how well the
               predicted box overlaps the ground-truth box
=======================================================================
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
from datetime import datetime

# YOLOv8 is provided by the 'ultralytics' package
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR    = PROJECT_ROOT / "2_model"
DATASET_YAML = MODEL_DIR   / "dataset.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"
WEIGHTS_DIR  = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING CONFIGURATION — Adjust these for your hardware
# ══════════════════════════════════════════════════════════════════════
TRAINING_CONFIG = {
    # ── Model ────────────────────────────────────────────────
    "model":     "yolov8m.pt",   # Pretrained weights (auto-downloaded)
    #             yolov8n.pt  → CPU / Raspberry Pi
    #             yolov8s.pt  → Laptop GPU
    #             yolov8m.pt  → Desktop GPU ✅ RECOMMENDED
    #             yolov8l.pt  → High-end GPU

    # ── Dataset ──────────────────────────────────────────────
    "data":      str(DATASET_YAML),

    # ── Training Duration ────────────────────────────────────
    "epochs":    100,             # Start with 50 for quick test, 100 for full
    "patience":  20,              # Early stopping: stop if no improvement after N epochs

    # ── Batch & Image Size ───────────────────────────────────
    "batch":     16,              # Reduce to 8 if you get CUDA out-of-memory
    "imgsz":     640,             # Input image size (YOLOv8 standard)

    # ── Hardware ─────────────────────────────────────────────
    "device":    "0",            # "0" = GPU 0,  "cpu" = CPU,  "0,1" = multi-GPU
    # "device":  "cpu",          # ← Uncomment this line if no GPU available

    # ── Optimisation ─────────────────────────────────────────
    "optimizer": "AdamW",         # SGD or AdamW
    "lr0":       0.001,           # Initial learning rate
    "lrf":       0.01,            # Final learning rate (fraction of lr0)
    "momentum":  0.937,
    "weight_decay": 0.0005,

    # ── Augmentation (helps prevent overfitting) ─────────────
    "hsv_h":     0.015,           # Hue shift
    "hsv_s":     0.7,             # Saturation shift
    "hsv_v":     0.4,             # Brightness shift
    "degrees":   5.0,             # Rotation (±5°)
    "translate": 0.1,             # Translation
    "scale":     0.5,             # Scale
    "shear":     2.0,             # Shear
    "perspective": 0.0,
    "flipud":    0.0,             # No vertical flip (upside-down roads are wrong!)
    "fliplr":    0.5,             # Horizontal flip (50% chance)
    "mosaic":    1.0,             # YOLOv8 mosaic augmentation (very effective!)
    "mixup":     0.1,             # Mixup augmentation

    # ── Output ───────────────────────────────────────────────
    "project":   str(RUNS_DIR),
    "name":      f"road_damage_{datetime.now().strftime('%Y%m%d_%H%M')}",
    "exist_ok":  True,
    "save":      True,
    "save_period": 10,            # Save checkpoint every 10 epochs
    "plots":     True,            # Save training plots
    "verbose":   True,
}


def check_dataset():
    """Verify that the dataset YAML and folders exist before training."""
    if not DATASET_YAML.exists():
        print(f"❌ dataset.yaml not found at: {DATASET_YAML}")
        print("   Run 1_dataset/download_datasets.py first!")
        sys.exit(1)

    with open(DATASET_YAML) as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg.get("path", "."))
    if not data_path.is_absolute():
        data_path = DATASET_YAML.parent / data_path

    train_img = data_path / cfg.get("train", "images/train")
    val_img   = data_path / cfg.get("val",   "images/val")

    if not train_img.exists():
        print(f"❌ Training images not found: {train_img}")
        sys.exit(1)

    n_train = len(list(train_img.glob("*.jpg"))) + len(list(train_img.glob("*.png")))
    n_val   = len(list(val_img.glob("*.jpg")))   + len(list(val_img.glob("*.png"))) if val_img.exists() else 0

    print(f"✅ Dataset found:")
    print(f"   Training images:   {n_train}")
    print(f"   Validation images: {n_val}")
    print(f"   Classes:           {cfg['names']}")


def train():
    """Main training function."""
    print("=" * 65)
    print("  AI Road Quality Monitor — YOLOv8 Training")
    print("=" * 65)

    # ── Pre-flight checks ─────────────────────────────────────────────
    check_dataset()

    # ── Check GPU ─────────────────────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"\n🎮 GPU detected: {gpu}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU detected — training on CPU (will be slow!)")
        print("   Recommendation: Use Google Colab (free T4 GPU)")
        print("   https://colab.research.google.com/")
        TRAINING_CONFIG["device"] = "cpu"
        TRAINING_CONFIG["batch"]  = 4
        TRAINING_CONFIG["epochs"] = 30   # Fewer epochs for CPU

    # ── Load pretrained YOLOv8 model ─────────────────────────────────
    print(f"\n📦 Loading model: {TRAINING_CONFIG['model']}")
    print("   (Pretrained on COCO — transfer learning will help a lot!)")
    model = YOLO(TRAINING_CONFIG["model"])

    # ── Start training ────────────────────────────────────────────────
    print(f"\n🚀 Starting training for {TRAINING_CONFIG['epochs']} epochs...")
    print(f"   Image size:  {TRAINING_CONFIG['imgsz']}×{TRAINING_CONFIG['imgsz']}")
    print(f"   Batch size:  {TRAINING_CONFIG['batch']}")
    print(f"   Results:     {RUNS_DIR}\n")

    results = model.train(**TRAINING_CONFIG)

    # ── Save best weights to /weights/ ────────────────────────────────
    run_dir   = Path(results.save_dir)
    best_pt   = run_dir / "weights" / "best.pt"
    last_pt   = run_dir / "weights" / "last.pt"

    if best_pt.exists():
        dest = WEIGHTS_DIR / "best.pt"
        shutil.copy(best_pt, dest)
        print(f"\n✅ Best weights saved: {dest}")

    return results, run_dir


def export_model(weights_path: Path = WEIGHTS_DIR / "best.pt", format: str = "onnx"):
    """
    Export the trained model for deployment.

    Formats:
      onnx      → Cross-platform (Windows/Linux/Mac/Edge)
      tflite    → Android/Raspberry Pi
      tensorrt  → NVIDIA Jetson / Server GPU (fastest)
      openvino  → Intel CPU (optimized)

    Usage:
      export_model(format="onnx")         # For PC deployment
      export_model(format="tflite")       # For mobile / Pi
      export_model(format="tensorrt")     # For NVIDIA Jetson
    """
    if not weights_path.exists():
        print(f"❌ Weights not found: {weights_path}. Train first!")
        return

    print(f"\n📤 Exporting model to {format.upper()} format...")
    model = YOLO(str(weights_path))

    # Export with hardware-specific optimizations
    export_args = {
        "format":    format,
        "imgsz":     640,
        "optimize":  True,   # Optimize for inference speed
        "simplify":  True,   # ONNX graph simplification
        "half":      True if format in ["engine", "tensorrt"] else False,
    }

    exported_path = model.export(**export_args)
    print(f"✅ Model exported to: {exported_path}")
    return exported_path


def print_training_tips():
    """Print tips for improving training results."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       TRAINING TIPS TO IMPROVE mAP                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 Data Quality (most important):                           ║
║    • More diverse images → better generalization             ║
║    • Consistent, tight bounding boxes                        ║
║    • Capture different: lighting, weather, road types        ║
║    • Balance classes (equal potholes, cracks, wear)          ║
║                                                              ║
║  🎛️  Hyperparameter Tuning:                                  ║
║    • Increase epochs if mAP still rising at epoch 100        ║
║    • Reduce lr0 if training is unstable                      ║
║    • Increase batch size for more stable gradients           ║
║    • Try yolov8l.pt if hardware allows                       ║
║                                                              ║
║  🔧 Advanced Tricks:                                         ║
║    • Use model.tune() for automatic hyperparameter search    ║
║    • Freeze backbone layers for tiny datasets                 ║
║    • Use CutMix or Mosaic augmentation (already enabled)     ║
║    • Fine-tune from RDD2022 pretrained weights               ║
║                                                              ║
║  📈 Typical mAP Benchmarks for Road Damage:                  ║
║    • 0.60–0.70 → Acceptable for proof of concept             ║
║    • 0.70–0.80 → Good – project quality                      ║
║    • 0.80–0.90 → Excellent – near state of the art           ║
║    • > 0.90    → Outstanding (hard to achieve)               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 for road damage detection")
    parser.add_argument("--train",  action="store_true", help="Run training")
    parser.add_argument("--export", action="store_true", help="Export trained model")
    parser.add_argument("--format", default="onnx",      help="Export format: onnx, tflite, tensorrt")
    parser.add_argument("--tips",   action="store_true", help="Show training tips")
    args = parser.parse_args()

    if args.tips:
        print_training_tips()
    elif args.export:
        export_model(format=args.format)
    else:
        # Default: train
        results, run_dir = train()
        print_training_tips()
        print(f"\n📂 All results saved to: {run_dir}")
        print(f"   View training plots:  {run_dir}/results.png")
        print(f"   Confusion matrix:     {run_dir}/confusion_matrix.png")
        print(f"\n▶  Next step: python 5_evaluation/evaluate_model.py")
