"""
Vision Model Wrapper - Classification Model
Handles loading and inference with EfficientNet/ResNet for Indian food classification
Optimized for Khana dataset (131K+ images, 80 classes)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import Dict, List, Optional, Union, Tuple
import os
import warnings
from pathlib import Path
import numpy as np
from PIL import Image
import math

# Indian food class names (matching nutrition database)
# These will be updated based on Khana dataset classes
INDIAN_FOOD_CLASSES = [
    "Biryani", "Dosa", "Idli", "Samosa", "Curry",
    "Naan", "Roti", "Dal", "Paneer Tikka", "Butter Chicken",
    "Palak Paneer", "Chole", "Rajma", "Aloo Gobi", "Baingan Bharta"
]

# Image preprocessing for classification models
CLASSIFICATION_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class VisionModel:
    """
    Wrapper class for classification-based food recognition
    Uses EfficientNet or ResNet for single-dish classification
    """
    
    def __init__(
        self,
        model_name="efficientnet_b0",
        model_path=None,
        num_classes=None,
        class_names=None,
        confidence_threshold=0.3,
        device=None,
        ood_entropy_threshold=2.0,
        ood_confidence_gap_threshold=0.3,
        ood_confidence_threshold=0.7
    ):
        """
        Initialize classification vision model
        
        Args:
            model_name (str): Model architecture (efficientnet_b0, resnet50, mobilenet_v2)
            model_path (str): Path to fine-tuned model weights (optional)
            num_classes (int): Number of classes (auto-detected from model if not provided)
            class_names (list): List of class names (default: INDIAN_FOOD_CLASSES)
            confidence_threshold (float): Minimum confidence for predictions
            device (str): Device to use ('cuda' or 'cpu')
            ood_entropy_threshold (float): Entropy threshold for OOD detection (default: 2.0)
            ood_confidence_gap_threshold (float): Confidence gap threshold for OOD detection (default: 0.3)
            ood_confidence_threshold (float): Confidence threshold for OOD detection (default: 0.7)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names if class_names is not None else INDIAN_FOOD_CLASSES
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.num_classes = num_classes
        self._is_loaded = False
        self.transform = CLASSIFICATION_TRANSFORM
        
        # OOD detection thresholds
        self.ood_entropy_threshold = ood_entropy_threshold
        self.ood_confidence_gap_threshold = ood_confidence_gap_threshold
        self.ood_confidence_threshold = ood_confidence_threshold
        
    def _create_model(self, num_classes: int):
        """Create model architecture"""
        if self.model_name.startswith("efficientnet"):
            # EfficientNet-B0, B1, B2, etc.
            variant = self.model_name.replace("efficientnet_", "").upper()
            try:
                weights = getattr(models, f"EfficientNet_{variant}_Weights").DEFAULT
                model = models.efficientnet_b0(weights=weights)
            except:
                # Fallback to B0
                weights = models.EfficientNet_B0_Weights.DEFAULT
                model = models.efficientnet_b0(weights=weights)
            
            # Replace classifier head
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, num_classes)
            )
            
        elif self.model_name.startswith("resnet"):
            # ResNet-50, ResNet-34, etc.
            if "50" in self.model_name:
                weights = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
            elif "34" in self.model_name:
                weights = models.ResNet34_Weights.DEFAULT
                model = models.resnet34(weights=weights)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
            else:
                # Default to ResNet-50
                weights = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
                
        elif self.model_name.startswith("mobilenet"):
            # MobileNet-V2
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Use efficientnet_b0, resnet50, or mobilenet_v2")
        
        return model
    
    def calculate_entropy(self, probabilities: torch.Tensor) -> float:
        """
        Calculate entropy of prediction distribution
        Higher entropy = more uncertain (predictions spread across classes)
        Lower entropy = more confident (one class dominates)
        
        Args:
            probabilities: Tensor of class probabilities [1, num_classes]
        
        Returns:
            float: Entropy value (higher = more uncertain)
        """
        # Remove zero probabilities to avoid log(0)
        probs = probabilities[0].cpu().numpy()
        probs = probs[probs > 1e-10]  # Filter out near-zero values
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def calculate_confidence_gap(self, probabilities: torch.Tensor) -> float:
        """
        Calculate the gap between top-1 and top-2 predictions
        Large gap = confident (one class clearly dominates)
        Small gap = uncertain (top predictions are close)
        
        Args:
            probabilities: Tensor of class probabilities [1, num_classes]
        
        Returns:
            float: Gap between top-1 and top-2 (0.0 to 1.0)
        """
        probs = probabilities[0].cpu().numpy()
        sorted_probs = np.sort(probs)[::-1]  # Sort descending
        
        if len(sorted_probs) < 2:
            return 1.0  # Only one class, maximum gap
        
        gap = float(sorted_probs[0] - sorted_probs[1])
        return gap
    
    def is_out_of_distribution(
        self, 
        confidence: float, 
        probabilities: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Detect if input is likely out-of-distribution (not an Indian dish)
        
        Uses multiple signals:
        1. Prediction entropy (high = uncertain)
        2. Confidence gap (small = uncertain)
        3. Overall confidence (low = uncertain)
        4. Suspiciously high confidence with low entropy (might be wrong prediction)
        
        Args:
            confidence: Top-1 prediction confidence
            probabilities: Full probability distribution [1, num_classes]
        
        Returns:
            tuple: (is_ood: bool, reason: str)
        """
        entropy = self.calculate_entropy(probabilities)
        confidence_gap = self.calculate_confidence_gap(probabilities)
        
        reasons = []
        is_ood = False
        
        # Check entropy (high entropy = uncertain)
        if entropy > self.ood_entropy_threshold:
            is_ood = True
            reasons.append(f"high uncertainty (entropy: {entropy:.2f})")
        
        # Check confidence gap (small gap = uncertain)
        if confidence_gap < self.ood_confidence_gap_threshold:
            is_ood = True
            reasons.append(f"close predictions (gap: {confidence_gap:.2f})")
        
        # Check overall confidence (low confidence = uncertain)
        if confidence < self.ood_confidence_threshold:
            is_ood = True
            reasons.append(f"low confidence ({confidence:.1%})")
        
        # Additional check: Very high confidence (>0.95) but entropy suggests uncertainty
        # This catches cases like eggs -> gulab jamun (high confidence but model is actually uncertain)
        # For 80 classes, normal high confidence should have entropy < 0.5
        # If entropy > 1.0 even with high confidence, it suggests uncertainty
        if confidence > 0.95 and entropy > 1.0:
            is_ood = True
            reasons.append(f"suspiciously high confidence with uncertainty (entropy: {entropy:.2f})")
        
        reason_str = ", ".join(reasons) if reasons else "normal prediction"
        
        return is_ood, reason_str
    
    def load_pretrained_model(self):
        """Load model (pretrained or fine-tuned)"""
        if self._is_loaded and self.model is not None:
            return  # Already loaded
        
        try:
            # Load fine-tuned model if available
            if self.model_path and os.path.exists(self.model_path):
                print(f"📥 Loading fine-tuned model from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        self.num_classes = checkpoint.get('num_classes', len(self.class_names))
                        if 'class_names' in checkpoint:
                            self.class_names = checkpoint['class_names']
                            print(f"✅ Loaded {len(self.class_names)} classes from checkpoint")
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                        self.num_classes = checkpoint.get('num_classes', len(self.class_names))
                    else:
                        state_dict = checkpoint
                        # Try to infer num_classes from state_dict
                        if self.num_classes is None:
                            # Get last layer size
                            for key in reversed(list(state_dict.keys())):
                                if 'classifier' in key or 'fc' in key:
                                    self.num_classes = state_dict[key].shape[0]
                                    break
                else:
                    state_dict = checkpoint
                
                if self.num_classes is None:
                    self.num_classes = len(self.class_names)
                
                # If class_names not loaded from checkpoint, try to load from class_names.txt
                if not self.class_names or len(self.class_names) != self.num_classes:
                    class_names_file = Path(self.model_path).parent / "class_names.txt"
                    if class_names_file.exists():
                        with open(class_names_file, 'r') as f:
                            self.class_names = [line.strip() for line in f if line.strip()]
                        print(f"✅ Loaded {len(self.class_names)} classes from {class_names_file}")
                
                self.model = self._create_model(self.num_classes)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                
            else:
                # Load pretrained model (ImageNet weights)
                print(f"📥 Loading pretrained {self.model_name} (ImageNet weights)")
                self.num_classes = len(self.class_names)
                self.model = self._create_model(self.num_classes)
                self.model.to(self.device)
                self.model.eval()
            
            self._is_loaded = True
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"   Classes: {self.num_classes}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict:
        """
        Predict food class from image
        
        Args:
            image: PIL Image, numpy array, or path to image file
        
        Returns:
            dict: Prediction results with food name and confidence
        """
        if self.model is None or not self._is_loaded:
            raise ValueError("Model not loaded. Call load_pretrained_model() first.")
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
        
        # Get top predictions
        top_k = min(5, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_predictions = []
        
        for prob, idx in zip(top_probs[0], top_indices[0]):
            if idx < len(self.class_names):
                top_predictions.append({
                    "food_name": self.class_names[idx],
                    "confidence": prob.item()
                })
        
        # Get predicted food name
        if predicted_idx < len(self.class_names):
            food_name = self.class_names[predicted_idx]
        else:
            food_name = "Unknown"
        
        # Check if confidence is too low
        is_uncertain = confidence < self.confidence_threshold
        
        # Detect out-of-distribution (OOD) inputs
        is_ood, ood_reason = self.is_out_of_distribution(confidence, probabilities)
        
        # Determine status
        status = "fine_tuned" if (self.model_path and os.path.exists(self.model_path)) else "pretrained"
        
        return {
            "food_name": food_name,
            "confidence": confidence,
            "class_index": predicted_idx,
            "top_predictions": top_predictions,
            "is_uncertain": is_uncertain,
            "is_out_of_distribution": is_ood,
            "ood_reason": ood_reason,
            "entropy": self.calculate_entropy(probabilities),
            "confidence_gap": self.calculate_confidence_gap(probabilities),
            "status": status,
            "model_name": self.model_name,
            "num_classes": self.num_classes
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path if self.model_path else "Pretrained (ImageNet)",
            "device": self.device,
            "num_classes": self.num_classes,
            "is_fine_tuned": self.model_path is not None and os.path.exists(self.model_path) if self.model_path else False,
            "is_loaded": self._is_loaded,
            "class_names": self.class_names[:10] if len(self.class_names) > 10 else self.class_names  # Show first 10
        }


def get_vision_model(
    model_name="efficientnet_b0",
    model_path=None,
    num_classes=None,
    class_names=None,
    confidence_threshold=0.3,
    ood_entropy_threshold=2.0, # Lower = more sensitive
    ood_confidence_gap_threshold=0.3, # Lower = more sensitive
    ood_confidence_threshold=0.7 # Higher = more sensitive
) -> VisionModel:
    """
    Factory function to create and load a vision model
    
    Args:
        model_name (str): Model architecture name
        model_path (str): Path to fine-tuned model
        num_classes (int): Number of classes
        class_names (list): List of class names
        confidence_threshold (float): Minimum confidence threshold
        ood_entropy_threshold (float): Entropy threshold for OOD detection (default: 2.0)
        ood_confidence_gap_threshold (float): Confidence gap threshold for OOD detection (default: 0.3)
        ood_confidence_threshold (float): Confidence threshold for OOD detection (default: 0.7)
    
    Returns:
        VisionModel: Loaded and ready-to-use vision model
    """
    model = VisionModel(
        model_name=model_name,
        model_path=model_path,
        num_classes=num_classes,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        ood_entropy_threshold=ood_entropy_threshold,
        ood_confidence_gap_threshold=ood_confidence_gap_threshold,
        ood_confidence_threshold=ood_confidence_threshold
    )
    model.load_pretrained_model()
    return model
