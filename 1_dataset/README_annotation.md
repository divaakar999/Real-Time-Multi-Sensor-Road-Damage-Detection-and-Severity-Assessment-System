# Roboflow Annotation Guide for Road Damage Detection

## 📌 Step-by-Step: Annotate Your Road Images with Roboflow

Roboflow is a free tool that lets you upload, label, and export datasets
in the exact format YOLOv8 expects. It also handles train/val/test splits
and augmentation automatically.

---

## 🔧 Setup

1. **Create a Free Account** at [https://roboflow.com](https://roboflow.com)
2. Click **"Create New Project"**
3. Fill in:
   - **Project Name**: `Road-Damage-Detection`
   - **License**: Choose based on your use
   - **Project Type**: `Object Detection`
   - **Annotation Group**: `road damage`
4. Click **"Create Project"**

---

## 📤 Upload Images

1. Drag & drop your collected images from `data/raw/local_collection/images/`
2. Roboflow will upload and display thumbnails
3. Click **"Finish Uploading"**

---

## 🏷️ Annotation Labels

Create exactly these class labels (match your `dataset.yaml`):

| Class Name   | Description                                       | Colour  |
|--------------|---------------------------------------------------|---------|
| `pothole`    | Potholes, holes in road surface                   | 🔴 Red  |
| `crack`      | All types: longitudinal, transverse, alligator    | 🟡 Yellow |
| `wear`       | Surface wear, raveling, patching                  | 🟠 Orange |

---

## ✏️ Annotation Best Practices

### Bounding Box Tips
- **Tight but complete**: Draw the box to exactly fit the damage boundary
- **Include context**: For cracks, include the full extent (don't cut off ends)
- **Overlapping OK**: Multiple damage labels can overlap on one image
- **Uncertain? Skip it**: Don't label damage you're unsure about

### What to Label vs Skip
✅ **Label these:**
- Clear potholes visible from any angle
- Visible crack patterns (even small ones)
- Patched areas with visible wear
- Water-filled potholes (the hole is still there)

❌ **Skip these:**
- Heavily blurred or dark images
- Speed bumps (speed bumps ≠ damage)
- Road markings or paint
- Shadows that look like damage

---

## 🔄 Keyboard Shortcuts in Roboflow Annotator

| Key       | Action                |
|-----------|-----------------------|
| `W`       | Select bounding box tool |
| `Q`       | Polygon tool          |
| `D`       | Next image            |
| `A`       | Previous image        |
| `Del`     | Delete selected label |
| `Ctrl+Z`  | Undo                  |
| `Space`   | Skip image            |

---

## 🔧 Pre-Processing Settings (Recommended)

After annotating, click **"Generate"** and configure:

**Pre-processing:**
- ✅ Auto-Orient (fix phone rotation)
- ✅ Resize: 640 × 640 (YOLOv8 standard)
- ✅ Grayscale: No (keep color)

**Augmentation (Roboflow handles this automatically):**
- ✅ Flip: Horizontal only (road damage is symmetric horizontally)
- ✅ Rotation: ±15°
- ✅ Brightness: -25% to +25%
- ✅ Blur: Up to 1.5px (simulates motion)
- ✅ Noise: Up to 5%
- ❌ Flip Vertical: No (upside-down roads don't make sense!)

**Split Ratio:**
- Train: 70%
- Validation: 20%
- Test: 10%

---

## 📥 Export Dataset

1. Click **"Export Dataset"**
2. Choose format: **"YOLOv8"**
3. Select **"download zip to computer"** OR **"Show Download Code"**
4. If downloading code:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("road-damage-detection")
version = project.version(1)
dataset = version.download("yolov8")
```

5. Extract to: `data/processed/`

---

## 📁 Expected Output Structure

After export, your `data/processed/` folder should look like:
```
data/processed/
├── data.yaml              ← dataset config (rename to dataset.yaml)
├── images/
│   ├── train/             ← ~70% of images
│   ├── valid/             ← ~20% of images
│   └── test/              ← ~10% of images
└── labels/
    ├── train/             ← YOLO format .txt files
    ├── valid/
    └── test/
```

---

## 🆓 Free Alternatives to Roboflow (Offline)

### CVAT (Computer Vision Annotation Tool)
```bash
# Run locally with Docker
docker compose -f docker-compose.yml up -d
# Visit: http://localhost:8080
```

### LabelImg (Simple Desktop App)
```bash
pip install labelImg
labelImg                   # Opens GUI
```
- Set "Save Dir" to `data/processed/labels/train`
- Set format to **YOLO**
- Use keyboard shortcut `W` to draw bounding boxes

### Label Studio
```bash
pip install label-studio
label-studio              # Opens at http://localhost:8080
```

---

## 📊 How Many Images Do You Need?

| Accuracy Target | Min Images (per class) | Total Dataset |
|-----------------|------------------------|---------------|
| Proof of Concept | 100 | 300–500 |
| Good (mAP > 0.7) | 500 | 1500–3000 |
| Production Ready | 2000+ | 6000–10000 |

**Tip**: Start with 200–300 well-annotated images. Fine-tune after seeing results.
You can also use pre-trained weights from RDD2022 and fine-tune on your local images.
