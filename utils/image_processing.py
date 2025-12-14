"""
Image Processing Utilities
Functions for preprocessing images before model inference
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def get_image_transform(input_size=224):
    """
    Get image transformation pipeline for EfficientNet
    
    Args:
        input_size (int): Target image size (default: 224 for EfficientNet)
    
    Returns:
        transforms.Compose: Transformation pipeline
    """
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def preprocess_image(image, input_size=224):
    """
    Preprocess image for model inference
    
    Args:
        image: PIL Image or numpy array
        input_size (int): Target image size
    
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL Image or numpy array")
    
    # Get transformation
    transform = get_image_transform(input_size)
    
    # Apply transformation
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def load_image_from_path(image_path):
    """
    Load image from file path
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        PIL.Image: Loaded image
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Error loading image from {image_path}: {str(e)}")


def validate_image(image):
    """
    Validate image format and size
    
    Args:
        image: PIL Image
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, Image.Image):
        return False, "Image must be PIL Image"
    
    # Check image size (minimum requirements)
    width, height = image.size
    if width < 32 or height < 32:
        return False, f"Image too small: {width}x{height} (minimum 32x32)"
    
    if width > 10000 or height > 10000:
        return False, f"Image too large: {width}x{height} (maximum 10000x10000)"
    
    return True, "Valid image"

