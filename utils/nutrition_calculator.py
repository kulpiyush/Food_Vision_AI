"""
Nutrition Calculator Utilities
Functions for looking up and calculating nutritional information
Uses Kaggle Indian Food Nutrition dataset directly with semantic matching
"""

import pandas as pd
import os
from pathlib import Path
import numpy as np
import pickle

# Try to use sentence transformers for semantic matching
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    USE_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

# Global cache for model and embeddings
_model_cache = None
_embeddings_cache = None
_kaggle_df_cache = None
_column_mapping_cache = None


def normalize_food_name(name):
    """Normalize food name for matching"""
    if pd.isna(name):
        return ""
    return str(name).lower().strip()


def load_kaggle_dataset(kaggle_path="data/Indian_Food_Nutrition_Processed.csv"):
    """
    Load and prepare Kaggle dataset with column mapping
    
    Args:
        kaggle_path (str): Path to Kaggle CSV file
    
    Returns:
        tuple: (DataFrame, dish_column_name, column_mapping_dict)
    """
    global _kaggle_df_cache, _column_mapping_cache
    
    # Use cache if available
    if _kaggle_df_cache is not None:
        return _kaggle_df_cache, _column_mapping_cache['dish_col'], _column_mapping_cache
    
    if not os.path.exists(kaggle_path):
        raise FileNotFoundError(f"Kaggle dataset not found at: {kaggle_path}")
    
    df = pd.read_csv(kaggle_path)
    
    # Find dish column
    dish_col = None
    for col in df.columns:
        if 'dish' in col.lower() or 'name' in col.lower():
            dish_col = col
            break
    
    if dish_col is None:
        raise ValueError("Could not find dish name column in Kaggle dataset")
    
    # Map column names to our standard format
    col_map = {
        'dish_col': dish_col,
        'calories': None,
        'carbs_g': None,
        'protein_g': None,
        'fat_g': None,
        'fiber_g': None
    }
    
    for col in df.columns:
        col_lower = col.lower()
        if 'calorie' in col_lower:
            col_map['calories'] = col
        elif 'carb' in col_lower:
            col_map['carbs_g'] = col
        elif 'protein' in col_lower:
            col_map['protein_g'] = col
        elif 'fat' in col_lower and 'free' not in col_lower:
            col_map['fat_g'] = col
        elif 'fibr' in col_lower:
            col_map['fiber_g'] = col
    
    # Cache
    _kaggle_df_cache = df
    _column_mapping_cache = col_map
    
    return df, dish_col, col_map


def get_sentence_transformer_model():
    """Get or load sentence transformer model (cached)"""
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    if not USE_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers not installed")
    
    # Try multilingual model first, fallback to English
    try:
        _model_cache = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except:
        _model_cache = SentenceTransformer('all-MiniLM-L6-v2')
    
    return _model_cache


def get_kaggle_embeddings(kaggle_df, dish_col):
    """Get or compute embeddings for Kaggle dataset (cached)"""
    global _embeddings_cache
    
    # Check cache file first
    cache_file = Path("data/.kaggle_embeddings_cache.pkl")
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data.get('dish_names_hash') == hash(tuple(kaggle_df[dish_col].astype(str))):
                    _embeddings_cache = cached_data['embeddings']
                    return _embeddings_cache
        except:
            pass
    
    # Compute embeddings
    if _embeddings_cache is not None:
        return _embeddings_cache
    
    model = get_sentence_transformer_model()
    kaggle_dish_names = [normalize_food_name(str(row[dish_col])) for _, row in kaggle_df.iterrows()]
    
    _embeddings_cache = model.encode(kaggle_dish_names, show_progress_bar=False, batch_size=32)
    
    # Save to cache file
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'embeddings': _embeddings_cache,
                'dish_names_hash': hash(tuple(kaggle_df[dish_col].astype(str)))
            }, f)
    except:
        pass
    
    return _embeddings_cache


def search_kaggle_with_semantic(food_name, kaggle_df, dish_col, col_map, threshold=0.5):
    """
    Search Kaggle dataset using semantic similarity
    
    Args:
        food_name (str): Food name to search for
        kaggle_df (pd.DataFrame): Kaggle dataset
        dish_col (str): Name of dish column
        col_map (dict): Column mapping dictionary
        threshold (float): Minimum similarity score
    
    Returns:
        tuple: (matching_row, similarity_score) or (None, best_score)
    """
    if not USE_SENTENCE_TRANSFORMERS:
        # Fallback to simple string matching
        food_normalized = normalize_food_name(food_name)
        mask = kaggle_df[dish_col].astype(str).str.lower().str.contains(food_normalized, na=False)
        matches = kaggle_df[mask]
        if not matches.empty:
            return matches.iloc[0], 0.8  # Approximate score
        return None, 0.0
    
    # Get embeddings
    kaggle_embeddings = get_kaggle_embeddings(kaggle_df, dish_col)
    model = get_sentence_transformer_model()
    
    # Encode query
    food_normalized = normalize_food_name(food_name)
    query_embedding = model.encode([food_normalized], show_progress_bar=False)
    
    # Calculate similarity
    similarities = cosine_similarity(query_embedding, kaggle_embeddings)[0]
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    if best_score >= threshold:
        return kaggle_df.iloc[best_idx], best_score
    
    return None, best_score


def extract_nutrition_from_kaggle(row, col_map):
    """Extract nutrition data from Kaggle row"""
    dish_col = col_map['dish_col']
    
    nutrition = {
        'food_name': str(row[dish_col]),
        'calories': float(row[col_map['calories']]) if col_map.get('calories') else 0,
        'fat_g': float(row[col_map['fat_g']]) if col_map.get('fat_g') else 0,
        'carbs_g': float(row[col_map['carbs_g']]) if col_map.get('carbs_g') else 0,
        'protein_g': float(row[col_map['protein_g']]) if col_map.get('protein_g') else 0,
        'fiber_g': float(row[col_map['fiber_g']]) if col_map.get('fiber_g') else 0,
        'per_100g': 100  # Kaggle data is per 100g
    }
    return nutrition


def get_nutrition(food_name, kaggle_path="data/Indian_Food_Nutrition_Processed.csv", portion_size=100, similarity_threshold=0.5):
    """
    Get nutritional information for a food item from Kaggle dataset using semantic matching
    
    Args:
        food_name (str): Name of the food (from model output)
        kaggle_path (str): Path to Kaggle dataset CSV
        portion_size (float): Portion size in grams (default: 100g)
        similarity_threshold (float): Minimum similarity score (default: 0.5)
    
    Returns:
        dict: Nutritional information, or None if not found
    """
    # Load Kaggle dataset
    try:
        kaggle_df, dish_col, col_map = load_kaggle_dataset(kaggle_path)
    except Exception as e:
        print(f"⚠️  Error loading Kaggle dataset: {e}")
        return None
    
    # Search using semantic similarity
    match, score = search_kaggle_with_semantic(food_name, kaggle_df, dish_col, col_map, threshold=similarity_threshold)
    
    if match is None:
        return None
    
    # Extract nutrition data
    nutrition = extract_nutrition_from_kaggle(match, col_map)
    
    # Calculate nutrition based on portion size
    scale_factor = portion_size / nutrition['per_100g']
    
    # Convert numpy types to Python native types
    similarity_score = float(score) if hasattr(score, 'item') else float(score)
    
    result = {
        'food_name': food_name,  # Use original name from model
        'matched_name': nutrition['food_name'],  # Name from Kaggle dataset
        'similarity_score': round(similarity_score, 2),
        'portion_size_g': portion_size,
        'calories': round(nutrition['calories'] * scale_factor, 1),
        'fat_g': round(nutrition['fat_g'] * scale_factor, 2),
        'carbs_g': round(nutrition['carbs_g'] * scale_factor, 2),
        'protein_g': round(nutrition['protein_g'] * scale_factor, 2),
        'fiber_g': round(nutrition['fiber_g'] * scale_factor, 2),
    }
    
    return result


def clear_cache():
    """Clear all caches (useful for testing or reloading data)"""
    global _model_cache, _embeddings_cache, _kaggle_df_cache, _column_mapping_cache
    _model_cache = None
    _embeddings_cache = None
    _kaggle_df_cache = None
    _column_mapping_cache = None
    
    # Remove cache file
    cache_file = Path("data/.kaggle_embeddings_cache.pkl")
    if cache_file.exists():
        cache_file.unlink()


