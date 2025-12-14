# Terminal Training Commands (Quick Reference)

## For JupyterLab Server

### Option 1: Use the Notebook ✅
1. Upload `train_yolo.ipynb` to server
2. Upload `data/yolo_training_data/` folder
3. Open notebook and run all cells

### Option 2: Use Terminal Commands

#### Step 1: Upload Dataset (from local machine)
```bash
# Upload the prepared dataset folder
scp -r data/yolo_training_data user@server:/path/to/project/
```

#### Step 2: SSH to Server
```bash
ssh user@server
cd /path/to/project
```

#### Step 3: Install Dependencies
```bash
pip install ultralytics
```

#### Step 4: Train Model
```bash
# Basic training
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(
    data='data/yolo_training_data/dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    project='models/weights',
    name='food_detector'
)
"

# Or use the training script (if uploaded)
python scripts/train_yolo_model.py --epochs 100 --batch 16
```

#### Step 5: Download Model
```bash
# From your local machine
scp user@server:/path/to/models/weights/food_detector/weights/best.pt ./food_detector_yolo.pt
```

---

## Quick One-Liner (If you have the script)

```bash
# On server
python scripts/train_yolo_model.py --epochs 100 --batch 16 --model yolov8n
```

---

## What to Upload to Server

**Minimum required:**
- `data/yolo_training_data/` folder (51 MB)
  - Contains: train/, val/, dataset.yaml

**Optional:**
- `train_yolo.ipynb` (if using notebook)
- `scripts/train_yolo_model.py` (if using terminal)

---

## Expected Output

After training, you'll find:
- `models/weights/food_detector/weights/best.pt` - Best model
- `models/weights/food_detector_yolo.pt` - Copied model (if script used)

Download the `.pt` file and place it in your local project!

