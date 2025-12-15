"""
Models package for FoodVisionAI
Contains vision models and generative AI models
"""

from .vision_model import VisionModel, get_vision_model, INDIAN_FOOD_CLASSES
from .genai_model import GenAIModel, get_genai_model

__all__ = [
    'VisionModel',
    'get_vision_model',
    'INDIAN_FOOD_CLASSES',
    'GenAIModel',
    'get_genai_model'
]

