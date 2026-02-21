<<<<<<< HEAD
# 🛣️ AI-Based Road Quality Monitoring System

A complete final-year project for detecting potholes, cracks, and road surface wear using YOLOv8 + GPS tagging + Streamlit dashboard.

## 📁 Project Structure

```
road_quality_monitor/
│
├── 1_dataset/
│   ├── download_datasets.py        # Download & prepare open-source datasets
│   ├── collect_local_images.py     # Capture road images from webcam/dashcam
│   └── README_annotation.md        # Guide to annotating with Roboflow
│
├── 2_model/
│   ├── train_yolov8.py             # Full YOLOv8 training pipeline
│   ├── evaluate_model.py           # mAP, precision, recall evaluation
│   └── dataset.yaml                # Dataset config for YOLO training
│
├── 3_detection/
│   ├── realtime_detection.py       # Live webcam/dashcam inference
│   ├── gps_tagger.py               # GPS coordinate tagging module
│   └── severity_classifier.py      # Classify damage severity
│
├── 4_dashboard/
│   ├── app.py                      # Main Streamlit dashboard
│   ├── map_component.py            # Folium map integration
│   ├── report_generator.py         # PDF/CSV report generation
│   └── assets/                     # Static files (CSS, icons)
│
├── 5_evaluation/
│   ├── benchmark.py                # Compare with baseline methods
│   └── visualize_metrics.py        # Plot training curves, confusion matrix
│
├── requirements.txt                # All Python dependencies
└── run.py                          # One-click launcher
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python 1_dataset/download_datasets.py

# 3. Train YOLOv8
python 2_model/train_yolov8.py

# 4. Launch Dashboard
streamlit run 4_dashboard/app.py
```

## 🎯 Detection Classes
- `pothole` - Potholes in road surface
- `crack` - Surface cracks (longitudinal / transverse / alligator)
- `wear` - Surface wear / raveling

## 🌡️ Severity Levels
| Level | Color | Criteria |
|-------|-------|----------|
| Low | 🟢 Green | Damage area < 5% of bounding box |
| Medium | 🟡 Yellow | Damage area 5–20% |
| High | 🔴 Red | Damage area > 20% |
=======
# Real-Time-Multi-Sensor-Road-Damage-Detection-and-Severity-Assessment-System
>>>>>>> e01f3cc01168d42d2be786986dc0904effefbda4
