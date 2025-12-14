"""
Vision Model Wrapper
Handles loading and inference with YOLO for multi-food detection
Phase 2: YOLO-based multi-food detection implementation
"""

import torch
from typing import Dict, List, Tuple, Optional, Union
import os
import warnings
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

# Indian food class names (matching nutrition database)
INDIAN_FOOD_CLASSES = [
    "Biryani", "Dosa", "Idli", "Samosa", "Curry",
    "Naan", "Roti", "Dal", "Paneer Tikka", "Butter Chicken",
    "Palak Paneer", "Chole", "Rajma", "Aloo Gobi", "Baingan Bharta"
]

# Mapping from YOLO dataset classes to our food classes
# Based on dataset.yaml from HuggingFace dataset
YOLO_TO_FOOD_MAPPING = {
    # YOLO class index -> Our food classes (can map to multiple)
    0: ["Naan", "Roti"],  # bread_or_Roti_naan
    1: ["Curry", "Butter Chicken", "Palak Paneer"],  # curry_dish
    2: ["Biryani"],  # rice_dish
    3: ["Aloo Gobi", "Baingan Bharta"],  # dry_vegetable
    4: ["Samosa"],  # snack_item
    7: ["Dal"],  # Dal_or_sambar
    15: ["Dosa", "Idli"],  # south_indian_breakfast
}

# Reverse mapping for easier lookup
FOOD_TO_YOLO_MAPPING = {}
for yolo_class, foods in YOLO_TO_FOOD_MAPPING.items():
    for food in foods:
        if food not in FOOD_TO_YOLO_MAPPING:
            FOOD_TO_YOLO_MAPPING[food] = []
        FOOD_TO_YOLO_MAPPING[food].append(yolo_class)


class VisionModel:
    """
    Wrapper class for YOLO-based multi-food detection
    Phase 2: YOLO implementation for detecting multiple foods per image
    """
    
    def __init__(
        self,
        model_name="yolov8n",
        model_path=None,
        dataset_yaml=None,
        confidence_threshold=0.25,
        class_names=None
    ):
        """
        Initialize YOLO vision model
        
        Args:
            model_name (str): YOLO model name (yolov8n, yolov8s, yolov8m, etc.)
            model_path (str): Path to fine-tuned YOLO model weights (optional)
            dataset_yaml (str): Path to dataset.yaml file (for class names)
            confidence_threshold (float): Confidence threshold for detections
            class_names (list): List of food class names (default: INDIAN_FOOD_CLASSES)
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        self.model_name = model_name
        self.model_path = model_path
        self.dataset_yaml = dataset_yaml or "data/hf_dataset/dataset.yaml"
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names if class_names is not None else INDIAN_FOOD_CLASSES
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded = False
        self.yolo_class_names = []  # Will be loaded from dataset or model
        
    def load_pretrained_model(self):
        """
        Load YOLO model (pretrained or fine-tuned)
        """
        if self._is_loaded and self.model is not None:
            return  # Already loaded
        
        try:
            # Load fine-tuned model if available
            if self.model_path and os.path.exists(self.model_path):
                print(f"📥 Loading fine-tuned YOLO model from {self.model_path}")
                self.model = YOLO(self.model_path)
                print(f"✅ Loaded fine-tuned model")
            else:
                # Load pretrained YOLO model
                print(f"📥 Loading pretrained {self.model_name} model...")
                self.model = YOLO(f"{self.model_name}.pt")
                print(f"✅ Loaded pretrained {self.model_name}")
            
            # Get class names from model or dataset
            if hasattr(self.model, 'names'):
                self.yolo_class_names = list(self.model.names.values())
            elif os.path.exists(self.dataset_yaml):
                import yaml
                with open(self.dataset_yaml, 'r') as f:
                    config = yaml.safe_load(f)
                    self.yolo_class_names = list(config.get('names', {}).values())
            
            self._is_loaded = True
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"   YOLO classes: {len(self.yolo_class_names)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    
    def _map_yolo_to_food(self, yolo_class_id: int, confidence: float) -> List[Dict]:
        """
        Map YOLO class detection to our food classes
        
        Args:
            yolo_class_id: YOLO class ID
            confidence: Detection confidence
        
        Returns:
            List of food detections with mapped classes
        """
        foods = YOLO_TO_FOOD_MAPPING.get(yolo_class_id, [])
        
        if not foods:
            # Unknown YOLO class, skip or use generic name
            yolo_name = self.yolo_class_names[yolo_class_id] if yolo_class_id < len(self.yolo_class_names) else f"Class_{yolo_class_id}"
            return [{
                "food_name": yolo_name,
                "confidence": confidence,
                "yolo_class": yolo_class_id,
                "mapped": False
            }]
        
        # Map to our food classes
        detections = []
        for food in foods:
            detections.append({
                "food_name": food,
                "confidence": confidence,
                "yolo_class": yolo_class_id,
                "mapped": True
            })
        
        return detections
    
    def predict(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict:
        """
        Predict food items from image (multi-food detection)
        
        Args:
            image: PIL Image, numpy array, or path to image file
        
        Returns:
            dict: Prediction results with multiple foods detected
        """
        if self.model is None or not self._is_loaded:
            raise ValueError("Model not loaded. Call load_pretrained_model() first.")
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run YOLO inference
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        
        # Process results
        all_detections = []
        
        if results and len(results) > 0:
            result = results[0]  # First (and usually only) result
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                # Get all detections
                for i in range(len(boxes)):
                    # Get class ID and confidence
                    class_id = int(boxes.cls[i].item())
                    confidence = float(boxes.conf[i].item())
                    
                    # Get bounding box
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    # Map YOLO class to our food classes
                    food_detections = self._map_yolo_to_food(class_id, confidence)
                    
                    # Add bounding box info to each detection
                    for food_det in food_detections:
                        food_det["bbox"] = bbox.tolist()
                        all_detections.append(food_det)
        
        # If no detections, return empty result
        if not all_detections:
            return {
                "foods": [],
                "food_name": "Unknown",
                "confidence": 0.0,
                "top_predictions": [],
                "status": "yolo_pretrained" if not self.model_path else "yolo_fine_tuned",
                "model_name": self.model_name,
                "num_detections": 0
            }
        
        # Sort by confidence
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Get primary detection (highest confidence)
        primary = all_detections[0]
        
        # Get top predictions (unique foods, highest confidence for each)
        food_confidences = {}
        for det in all_detections:
            food_name = det["food_name"]
            if food_name not in food_confidences or det["confidence"] > food_confidences[food_name]["confidence"]:
                food_confidences[food_name] = det
        
        top_predictions = sorted(
            food_confidences.values(),
            key=lambda x: x["confidence"],
            reverse=True
        )[:5]
        
        # Determine status
        status = "yolo_fine_tuned" if (self.model_path and os.path.exists(self.model_path)) else "yolo_pretrained"
        
        return {
            "food_name": primary["food_name"],
            "confidence": primary["confidence"],
            "top_predictions": [
                {"food_name": pred["food_name"], "confidence": pred["confidence"]}
                for pred in top_predictions
            ],
            "foods": all_detections,  # All detected foods (multi-food support)
            "status": status,
            "model_name": self.model_name,
            "num_detections": len(all_detections)
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "is_fine_tuned": self.model_path is not None and os.path.exists(self.model_path) if self.model_path else False,
            "confidence_threshold": self.confidence_threshold,
            "yolo_classes": len(self.yolo_class_names),
            "food_classes": len(self.class_names)
        }


def get_vision_model(
    model_name="yolov8n",
    model_path=None,
    dataset_yaml=None
) -> VisionModel:
    """
    Factory function to create and load a YOLO vision model
    
    Args:
        model_name (str): YOLO model name
        model_path (str): Path to fine-tuned weights (optional)
        dataset_yaml (str): Path to dataset.yaml (optional)
    
    Returns:
        VisionModel: Loaded and ready-to-use vision model
    """
    model = VisionModel(
        model_name=model_name,
        model_path=model_path,
        dataset_yaml=dataset_yaml
    )
    model.load_pretrained_model()
    return model


# Keep old functions for backward compatibility (if needed)
def create_placeholder_prediction():
    """DEPRECATED: Use VisionModel.predict() instead"""
    import random
    food = random.choice(INDIAN_FOOD_CLASSES)
    
    return {
        "food_name": food,
        "confidence": 0.85 + random.random() * 0.1,
        "top_predictions": [
            {"food_name": food, "confidence": 0.85 + random.random() * 0.1},
            {"food_name": random.choice(INDIAN_FOOD_CLASSES), "confidence": 0.5 + random.random() * 0.2},
            {"food_name": random.choice(INDIAN_FOOD_CLASSES), "confidence": 0.3 + random.random() * 0.2},
        ],
        "status": "placeholder",
        "foods": [{"food_name": food, "confidence": 0.85 + random.random() * 0.1}],
        "num_detections": 1
    }
