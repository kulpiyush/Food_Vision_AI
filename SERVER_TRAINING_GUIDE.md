# Training on University Server - Complete Guide

## Overview

This guide helps you train the YOLO model on your university's JupyterLab server (faster with GPU).

## Option 1: Jupyter Notebook (Recommended) ✅

### Step 1: Upload Files to Server

**Upload these to your server:**
1. `train_yolo.ipynb` (the notebook I created)
2. `data/yolo_training_data/` folder (entire folder with train/val/images/labels)

**How to upload:**
- **JupyterLab**: Use file browser → Upload button
- **Terminal**: `scp -r data/yolo_training_data user@server:/path/to/project/`

### Step 2: Open and Run Notebook

1. Open `train_yolo.ipynb` in JupyterLab
2. Run cells sequentially
3. Training will start automatically
4. Model will be saved when complete

### Step 3: Download Trained Model

**In JupyterLab:**
- Right-click on `models/weights/food_detector_yolo.pt`
- Select "Download"

**Or use terminal:**
```bash
scp user@server:/path/to/models/weights/food_detector_yolo.pt ./
```

---

## Option 2: Terminal Commands

If you prefer terminal over notebook:

### Step 1: Upload Dataset

```bash
# From your local machine
scp -r data/yolo_training_data user@server:/path/to/project/
```

### Step 2: SSH to Server

```bash
ssh user@server
cd /path/to/project
```

### Step 3: Install Dependencies

```bash
pip install ultralytics
```

### Step 4: Train Model

```bash
# Basic training
python scripts/train_yolo_model.py

# Or with custom parameters
python scripts/train_yolo_model.py \
    --data data/yolo_training_data/dataset.yaml \
    --model yolov8n \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

### Step 5: Download Model

```bash
# From your local machine
scp user@server:/path/to/models/weights/food_detector_yolo.pt ./
```

---

## Quick Start (JupyterLab)

1. **Upload `train_yolo.ipynb`** to server
2. **Upload `data/yolo_training_data/`** folder
3. **Open notebook** in JupyterLab
4. **Run all cells** (Cell → Run All)
5. **Wait for training** (30-60 min on GPU)
6. **Download model** when done

---

## What to Upload

### Required Files:
```
📁 Your project on server should have:
├── train_yolo.ipynb                    # Training notebook
├── data/
│   └── yolo_training_data/             # Dataset folder
│       ├── train/
│       │   ├── images/  (301 images)
│       │   └── labels/  (301 labels)
│       ├── val/
│       │   ├── images/  (76 images)
│       │   └── labels/  (76 labels)
│       └── dataset.yaml
└── scripts/
    └── train_yolo_model.py             # (Optional, if using terminal)
```

### Files You DON'T Need:
- ❌ `app.py` (not needed for training)
- ❌ `models/vision_model.py` (not needed for training)
- ❌ Other project files (just dataset + notebook)

---

## Training Parameters

### Recommended Settings:

**For GPU (Fast):**
```python
MODEL_NAME = "yolov8n"  # or yolov8s for better accuracy
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
```

**For CPU (Slower):**
```python
MODEL_NAME = "yolov8n"  # Use smallest model
EPOCHS = 50  # Fewer epochs
BATCH_SIZE = 8  # Smaller batch
IMG_SIZE = 640
```

---

## Expected Training Time

| Hardware | Time |
|----------|------|
| GPU (CUDA) | 30-60 minutes |
| CPU | 3-6 hours |

---

## After Training

### 1. Download Model
- Model saved at: `models/weights/food_detector_yolo.pt`
- Size: ~6-12 MB

### 2. Place in Local Project
```bash
# On your local machine
cp food_detector_yolo.pt models/weights/
```

### 3. Update App (if needed)
The app will automatically detect and use the model if it's at:
```
models/weights/food_detector_yolo.pt
```

---

## Troubleshooting

### Issue: "Dataset not found"
**Solution:** Make sure `data/yolo_training_data/` folder is uploaded correctly

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size:
```python
BATCH_SIZE = 8  # or 4
```

### Issue: Training too slow
**Solution:** 
- Use GPU if available
- Use smaller model (yolov8n)
- Reduce epochs (50 instead of 100)

### Issue: Can't download model
**Solution:** Use terminal:
```bash
scp user@server:/path/to/models/weights/food_detector_yolo.pt ./
```

---

## Quick Reference

### Upload Dataset:
```bash
scp -r data/yolo_training_data user@server:/path/to/project/
```

### Train (Terminal):
```bash
python scripts/train_yolo_model.py --epochs 100 --batch 16
```

### Download Model:
```bash
scp user@server:/path/to/models/weights/food_detector_yolo.pt ./
```

---

**Ready to train!** Upload the notebook and dataset, then run it on your server! 🚀

