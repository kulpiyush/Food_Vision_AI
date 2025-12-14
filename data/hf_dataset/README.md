---
title: SOHL Multi-Dish Indian Food Detection Dataset
emoji: 🍽️
colorFrom: orange
colorTo: red
sdk: static
pinned: false
tags:
- computer-vision
- object-detection
- yolo
- food-detection
- indian-cuisine
- multi-dish
- yolov8
license: mit
---

# 🍽️ SOHL Multi-Dish Indian Food Detection Dataset

## Overview
This dataset contains **377 annotated images** of Indian food plates with **multiple dishes per image**. Designed for training YOLO models to detect and classify multiple food items on a single plate.

## Dataset Statistics
- **Images**: 377
- **Annotations**: 377  
- **Classes**: 16
- **Format**: YOLOv8 (images + txt annotations)
- **Created**: 2025-08-16

## Classes
0. **bread_or_Roti_naan** - Chapati, naan, roti, paratha, and other Indian breads
1. **curry_dish** - General curry preparations, gravies, and liquid dishes
2. **rice_dish** - Plain rice, biryani, pulao, and rice preparations
3. **dry_vegetable** - Bhindi, aloo, cauliflower, and dry sabzi preparations
4. **snack_item** - Samosa, pakora, vada, dhokla, and fried snacks
5. **sweet_item** - Traditional sweets, desserts, and mithai
6. **accompaniment** - Pickle, raita, papad, chutney, and side dishes
7. **Dal_or_sambar** - Dal preparations, sambar, and lentil-based dishes
8. **drink** - Beverages, juices, lassi, and liquid refreshments
9. **eggs** - Egg preparations, omelettes, and egg-based dishes
10. **fish_dish** - Fish curry, fried fish, and seafood preparations
11. **fruits** - Fresh fruits, fruit salads, and fruit-based items
12. **pasta** - Pasta dishes and Italian preparations
13. **salad** - Vegetable salads, mixed salads, and fresh preparations
14. **soup** - Soups, broths, and liquid appetizers
15. **south_indian_breakfast** - Dosa, idli, upma, and South Indian breakfast items

## Dataset Structure
```
sohl-multidish-yolo-dataset/
├── images/           # 377 image files
├── labels/           # 377 YOLO format annotations
├── dataset.yaml      # YOLOv8 configuration
└── README.md         # This file
```

## Usage

### Download Dataset
```python
from huggingface_hub import snapshot_download

# Download entire dataset
dataset_path = snapshot_download(
    repo_id="SohlHealth/sohl-multidish-yolo-dataset",
    repo_type="dataset"
)
```

### Train YOLOv8
```python
from ultralytics import YOLO

# Load model and train
model = YOLO('yolov8s.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    batch=8,
    imgsz=640
)
```

## Key Features
- ✅ **Multi-dish detection**: 2-6 items per plate
- ✅ **Indian cuisine focus**: Traditional dishes and combinations  
- ✅ **Real-world scenarios**: Restaurant and home environments
- ✅ **Complex layouts**: Overlapping items, various plate styles
- ✅ **High-quality annotations**: Precise bounding boxes
- ✅ **Comprehensive classes**: 16 food categories including regional specialties

## Performance Expectations
Based on similar datasets and architectures:
- **Expected mAP@0.5**: 15-25% (multi-dish detection is challenging)
- **Training time**: 3-6 hours on modern GPU
- **Recommended epochs**: 100-150
- **Best practices**: Transfer learning from food detection models

## Citation
```
@dataset{sohl_multidish_dataset_20250816_161951,
  title={SOHL Multi-Dish Indian Food Detection Dataset},
  author={SOHL AI Team},
  year={2025},
  url={https://huggingface.co/datasets/SohlHealth/sohl-multidish-yolo-dataset}
}
```

## License
MIT License - See LICENSE file for details.

## Contact
For questions about this dataset, please contact the SOHL AI team.
