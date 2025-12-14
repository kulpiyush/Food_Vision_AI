# How to Improve YOLO Model Accuracy

## Current Status
- **Dataset**: 377 images (301 train, 76 val)
- **Epochs**: 100
- **mAP50**: 77.6%
- **Issue**: Misclassifications (e.g., eggs → curry_dish)

## Methods to Improve Accuracy (Ranked by Impact)

### 🥇 **1. MORE TRAINING DATA (Highest Impact)**

**Why**: 377 images is very small for 16 classes (~23 images/class). More data = better accuracy.

**How to get more data:**
```bash
# Option A: Download more from HuggingFace
# Look for larger food detection datasets

# Option B: Collect your own images
# - Take photos of foods
# - Use food image datasets (Food-101, UEC-Food100, etc.)
# - Web scraping (with permission)

# Option C: Data augmentation (see below)
```

**Target**: Aim for **1000-2000+ images** (60-125 per class)

---

### 🥈 **2. Better Data Augmentation**

**Why**: Makes model more robust to variations (lighting, angle, background)

**Update training script:**
```python
results = model.train(
    data=str(data_path.absolute()),
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    # Add these augmentation parameters:
    hsv_h=0.02,      # Hue augmentation (default: 0.015)
    hsv_s=0.7,       # Saturation augmentation
    hsv_v=0.4,       # Value augmentation
    degrees=10.0,    # Rotation ±10 degrees (default: 0.0)
    translate=0.1,   # Translation (default: 0.1)
    scale=0.5,       # Scaling (default: 0.5)
    shear=5.0,       # Shearing ±5 degrees (default: 0.0)
    perspective=0.0, # Perspective transform
    flipud=0.0,      # Vertical flip probability
    fliplr=0.5,      # Horizontal flip (default: 0.5)
    mosaic=1.0,      # Mosaic augmentation (default: 1.0)
    mixup=0.1,       # Mixup augmentation (default: 0.0)
    copy_paste=0.0,  # Copy-paste augmentation
    project=output_dir,
    name="food_detector",
    save=True,
    plots=True,
    val=True
)
```

---

### 🥉 **3. More Epochs (But with Early Stopping)**

**Why**: 100 epochs might not be enough, but watch for overfitting.

**Update training:**
```python
results = model.train(
    data=str(data_path.absolute()),
    epochs=200,  # Increase from 100
    batch=batch_size,
    imgsz=img_size,
    patience=50,  # Early stopping: stop if no improvement for 50 epochs
    save_period=10,  # Save checkpoint every 10 epochs
    # ... other params
)
```

**Watch for**: Validation loss should decrease. If it increases, model is overfitting.

---

### 4. **Larger Model Architecture**

**Why**: YOLOv8n (nano) is smallest. Larger models = better accuracy.

**Options:**
- `yolov8s.pt` - Small (better accuracy, ~2x slower)
- `yolov8m.pt` - Medium (even better, ~4x slower)
- `yolov8l.pt` - Large (best, ~8x slower)

**Update training:**
```bash
python scripts/train_yolo_model.py --model yolov8s --epochs 150 --batch 16
```

---

### 5. **Better Hyperparameters**

**Learning Rate Tuning:**
```python
results = model.train(
    # ... other params
    lr0=0.001,      # Initial learning rate (default: 0.01)
    lrf=0.1,        # Final learning rate (lr0 * lrf)
    momentum=0.937, # SGD momentum
    weight_decay=0.0005,  # L2 regularization
    warmup_epochs=3.0,    # Warmup epochs
    warmup_momentum=0.8,   # Warmup momentum
    warmup_bias_lr=0.1,    # Warmup bias LR
)
```

**Optimizer:**
```python
optimizer='AdamW',  # Instead of 'auto' or 'SGD'
```

---

### 6. **Class Balancing**

**Problem**: Some classes might have very few images.

**Check class distribution:**
```python
# Count images per class in your dataset
import yaml
from pathlib import Path

dataset_yaml = Path("data/yolo_training_data/dataset.yaml")
with open(dataset_yaml, 'r') as f:
    config = yaml.safe_load(f)

# Check label files to count per class
# Classes with <10 images need more data
```

**Solution**: Collect more images for underrepresented classes.

---

### 7. **Image Size**

**Why**: Larger images = more detail = better accuracy (but slower)

**Update:**
```python
imgsz=1280,  # Instead of 640 (2x larger, 4x slower)
```

**Trade-off**: Better accuracy but much slower training/inference.

---

### 8. **Transfer Learning from Food-Specific Model**

**Why**: Start from a model already trained on food images.

**Options:**
- Use a pretrained food detection model
- Fine-tune from YOLOv8 trained on COCO (you're already doing this)

---

## Recommended Training Plan

### Phase 1: Quick Wins (Do First)
1. ✅ **Add data augmentation** (5 min to update script)
2. ✅ **Increase epochs to 150-200** with early stopping
3. ✅ **Try yolov8s** instead of yolov8n

### Phase 2: Data Collection (Most Important)
1. **Collect more images** - Aim for 1000+ total
2. **Balance classes** - At least 30-50 images per class
3. **Quality over quantity** - Clear, well-lit images

### Phase 3: Fine-tuning
1. **Tune hyperparameters** based on validation results
2. **Try larger image size** (if you have GPU)
3. **Ensemble models** (combine multiple models)

---

## Updated Training Script

I'll create an improved training script with all these optimizations:

```bash
# Train with better settings
python scripts/train_yolo_model.py \
    --model yolov8s \
    --epochs 200 \
    --batch 16 \
    --imgsz 640
```

---

## Expected Improvements

| Method | Expected mAP50 Gain | Effort |
|--------|---------------------|--------|
| More data (2x) | +5-10% | High |
| Data augmentation | +2-5% | Low |
| More epochs (200) | +1-3% | Low |
| Larger model (yolov8s) | +3-7% | Low |
| Better hyperparameters | +1-2% | Medium |
| Larger image size | +2-4% | Medium |

**Combined**: Could reach **85-90% mAP50** with all improvements!

---

## Quick Start: Retrain with Better Settings

1. **Update training script** with augmentation
2. **Increase epochs** to 200
3. **Try yolov8s** model
4. **Monitor validation metrics** during training

Would you like me to update the training script with these improvements?

