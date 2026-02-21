"""
=======================================================================
STEP 2B - MODEL: Evaluate YOLOv8 Model Performance
=======================================================================
Metrics computed:
  • mAP50       Mean Average Precision @ IoU=0.50
  • mAP50-95    Mean Average Precision averaged over IoU 0.50:0.95
  • Precision   How many detections were correct?
  • Recall      How many real damages were detected?
  • F1 Score    Harmonic mean of precision and recall
  • FPS         Inference frames per second
  • Confusion Matrix
  • Per-class AP breakdown
=======================================================================
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
WEIGHTS_PATH  = PROJECT_ROOT / "weights" / "best.pt"
DATASET_YAML  = PROJECT_ROOT / "2_model"  / "dataset.yaml"
EVAL_DIR      = PROJECT_ROOT / "evaluation_results"
EVAL_DIR.mkdir(exist_ok=True)

# Class names (must match dataset.yaml)
CLASS_NAMES   = ["pothole", "crack", "wear"]


def run_validation(weights_path: Path = WEIGHTS_PATH, split: str = "test"):
    """
    Run official YOLOv8 validation on the test set.

    Args:
        weights_path: Path to best.pt
        split:        "test", "val", or "train"

    Returns:
        metrics dict with all computed values
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating model on {split.upper()} set")
    print(f"{'='*60}")

    if not weights_path.exists():
        print(f"❌ Weights not found: {weights_path}")
        print("   Train the model first: python 2_model/train_yolov8.py")
        return None

    model   = YOLO(str(weights_path))

    # ── Run validation ─────────────────────────────────────────────────
    results = model.val(
        data    = str(DATASET_YAML),
        split   = split,
        imgsz   = 640,
        batch   = 16,
        conf    = 0.25,   # Confidence threshold
        iou     = 0.50,   # IoU threshold for matching
        plots   = True,   # Save confusion matrix, PR curve etc.
        save_json = True, # Save COCO-format results
        verbose = True,
    )

    # ── Extract metrics ────────────────────────────────────────────────
    metrics = {
        "mAP50":           float(results.box.map50),
        "mAP50_95":        float(results.box.map),
        "precision":       float(results.box.mp),
        "recall":          float(results.box.mr),
        "per_class_ap50":  {},
        "timestamp":       datetime.now().isoformat(),
    }

    # Per-class breakdown
    if hasattr(results.box, "ap50"):
        for i, cls_name in enumerate(CLASS_NAMES):
            if i < len(results.box.ap50):
                metrics["per_class_ap50"][cls_name] = float(results.box.ap50[i])

    # F1 Score
    p = metrics["precision"]
    r = metrics["recall"]
    metrics["f1"] = 2 * p * r / (p + r + 1e-10)

    # ── Print Summary ──────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  EVALUATION RESULTS ({split.upper()} SET)")
    print(f"{'─'*50}")
    print(f"  mAP50:      {metrics['mAP50']:.4f}  ({metrics['mAP50']*100:.1f}%)")
    print(f"  mAP50-95:   {metrics['mAP50_95']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"\n  Per-class AP50:")
    for cls, ap in metrics["per_class_ap50"].items():
        bar = "█" * int(ap * 20) + "░" * (20 - int(ap * 20))
        print(f"    {cls:10s} {bar} {ap:.3f}")

    # Save metrics JSON
    metrics_path = EVAL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved: {metrics_path}")

    return metrics


def benchmark_inference_speed(weights_path: Path = WEIGHTS_PATH, n_runs: int = 100):
    """
    Measure real-time capability by benchmarking inference speed.

    Returns FPS (frames per second).
    Target: > 25 FPS for real-time detection.
    """
    import torch
    from PIL import Image

    print(f"\n⏱️  Benchmarking inference speed ({n_runs} runs)...")

    model = YOLO(str(weights_path))

    # Warm up the model (first run is always slower)
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy_img, verbose=False)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(dummy_img, verbose=False)
        times.append(time.perf_counter() - start)

    avg_ms  = np.mean(times) * 1000
    std_ms  = np.std(times)  * 1000
    fps     = 1 / np.mean(times)

    print(f"  Average inference: {avg_ms:.1f} ms ± {std_ms:.1f} ms")
    print(f"  FPS:               {fps:.1f}")

    status = "✅ REAL-TIME CAPABLE" if fps > 25 else "⚠️  Below 25 FPS — consider smaller model"
    print(f"  Status:            {status}")

    return {"avg_ms": avg_ms, "std_ms": std_ms, "fps": fps}


def plot_metrics_dashboard(metrics: dict, speed: dict):
    """
    Generate a publication-quality metrics dashboard figure.
    Saved to evaluation_results/metrics_dashboard.png
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Road Damage Detection — Model Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")

    # ── 1. Main Metrics Bar Chart ──────────────────────────────────────
    ax1     = axes[0, 0]
    labels  = ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]
    values  = [
        metrics["mAP50"],
        metrics["mAP50_95"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    ]
    colors  = ["#00d4ff", "#00b894", "#fdcb6e", "#e17055", "#a29bfe"]
    bars    = ax1.barh(labels, values, color=colors, edgecolor="#30363d")
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel("Score", color="white")
    ax1.set_title("Overall Metrics", color="white", fontsize=12)
    ax1.tick_params(colors="white")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")
    for bar, val in zip(bars, values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color="white", fontsize=10)

    # ── 2. Per-class AP50 ──────────────────────────────────────────────
    ax2      = axes[0, 1]
    cls_names = list(metrics["per_class_ap50"].keys()) or CLASS_NAMES
    cls_vals  = list(metrics["per_class_ap50"].values()) or [0.0] * len(CLASS_NAMES)
    cls_colors = ["#e17055", "#fdcb6e", "#00cec9"]
    ax2.bar(cls_names, cls_vals, color=cls_colors[:len(cls_names)], edgecolor="#30363d")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("AP50", color="white")
    ax2.set_title("Per-Class AP50", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    for v, cls in zip(cls_vals, cls_names):
        ax2.text(cls, v + 0.02, f"{v:.3f}", ha="center", color="white", fontsize=10)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # ── 3. Precision-Recall trade-off (conceptual) ────────────────────
    ax3 = axes[0, 2]
    p   = metrics["precision"]
    r   = metrics["recall"]
    ax3.scatter([r], [p], color="#00d4ff", s=200, zorder=5)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Recall", color="white")
    ax3.set_ylabel("Precision", color="white")
    ax3.set_title("Precision vs Recall", color="white", fontsize=12)
    ax3.tick_params(colors="white")
    ax3.annotate(f"  F1={metrics['f1']:.3f}\n  P={p:.3f}\n  R={r:.3f}",
                 xy=(r, p), color="white", fontsize=9)
    ax3.plot([0, 1], [1, 0], "--", color="#30363d", alpha=0.5)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#30363d")

    # ── 4. Speed Gauge ────────────────────────────────────────────────
    ax4  = axes[1, 0]
    fps  = speed.get("fps", 0)
    gauge_colors = ["#e17055", "#fdcb6e", "#00b894"]
    thresholds   = [0, 15, 25, 60]
    theta        = [0, np.pi * 0.5, np.pi * 0.833, np.pi]
    for i in range(len(theta) - 1):
        wedge = mpatches.Wedge(
            [0.5, 0], 0.4,
            np.degrees(theta[i]),
            np.degrees(theta[i + 1]),
            width=0.1,
            color=gauge_colors[i],
        )
        ax4.add_patch(wedge)
    norm_fps  = min(fps / 60, 1.0)
    angle     = np.pi * (1 - norm_fps)
    ax4.annotate("", xy=(0.5 + 0.3 * np.cos(angle), 0 + 0.3 * np.sin(angle)),
                 xytext=(0.5, 0),
                 arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax4.text(0.5, -0.25, f"{fps:.1f} FPS", ha="center", va="center",
             color="white", fontsize=14, fontweight="bold")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_title("Inference Speed", color="white", fontsize=12)
    ax4.axis("off")

    # ── 5. Grade Summary ──────────────────────────────────────────────
    ax5 = axes[1, 1]
    ax5.axis("off")
    map50 = metrics["mAP50"]
    if map50 >= 0.90:
        grade, gcolor = "A+ Outstanding", "#00b894"
    elif map50 >= 0.80:
        grade, gcolor = "A  Excellent",   "#00cec9"
    elif map50 >= 0.70:
        grade, gcolor = "B  Good",        "#fdcb6e"
    elif map50 >= 0.60:
        grade, gcolor = "C  Acceptable",  "#e17055"
    else:
        grade, gcolor = "D  Needs Work",  "#d63031"

    ax5.text(0.5, 0.6, grade, ha="center", va="center",
             color=gcolor, fontsize=22, fontweight="bold",
             transform=ax5.transAxes)
    ax5.text(0.5, 0.35, f"mAP50 = {map50:.4f}", ha="center", va="center",
             color="white", fontsize=14, transform=ax5.transAxes)
    summary_text = (
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall:    {metrics['recall']:.3f}\n"
        f"F1 Score:  {metrics['f1']:.3f}\n"
        f"Speed:     {fps:.1f} FPS"
    )
    ax5.text(0.5, 0.05, summary_text, ha="center", va="center",
             color="#aaaaaa", fontsize=10, transform=ax5.transAxes,
             family="monospace")
    ax5.set_title("Model Grade", color="white", fontsize=12)

    # ── 6. Improvement Suggestions ────────────────────────────────────
    ax6  = axes[1, 2]
    ax6.axis("off")
    tips = [
        "✅ More diverse training data",
        "✅ Longer training (more epochs)",
        "✅ Larger model (yolov8l)",
        "✅ Better augmentation",
        "✅ Reduce class imbalance",
        "✅ Pre-train on RDD2022 first",
    ]
    ax6.text(0.05, 0.95, "Ways to Improve:", color="#00d4ff",
             va="top", transform=ax6.transAxes, fontsize=11, fontweight="bold")
    for i, tip in enumerate(tips):
        ax6.text(0.05, 0.80 - i * 0.13, tip, color="white",
                 va="top", transform=ax6.transAxes, fontsize=9)
    ax6.set_title("Optimisation Tips", color="white", fontsize=12)

    # ── Save ──────────────────────────────────────────────────────────
    plt.tight_layout()
    out_path = EVAL_DIR / "metrics_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n📊 Dashboard saved: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate road damage detection model")
    parser.add_argument("--weights", default=str(WEIGHTS_PATH), help="Path to best.pt")
    parser.add_argument("--split",   default="test", choices=["test", "val"])
    args  = parser.parse_args()

    weights = Path(args.weights)
    metrics = run_validation(weights, split=args.split)

    if metrics:
        speed   = benchmark_inference_speed(weights)
        metrics.update({"speed": speed})
        plot_metrics_dashboard(metrics, speed)

        print(f"\n✅ Evaluation complete!")
        print(f"   Results directory: {EVAL_DIR}")
