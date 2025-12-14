"""
Nutrition Calculator Utilities
Functions for looking up and calculating nutritional information
"""

import pandas as pd
import os
from pathlib import Path


def load_nutrition_database(db_path="data/nutrition_db.csv"):
    """
    Load nutritional database from CSV file
    
    Args:
        db_path (str): Path to nutrition database CSV
    
    Returns:
        pd.DataFrame: Nutrition database
    """
    if not os.path.exists(db_path):
        # Return empty DataFrame if file doesn't exist yet
        return pd.DataFrame(columns=[
            'food_name', 'calories', 'fat_g', 'carbs_g', 
            'protein_g', 'fiber_g', 'per_100g'
        ])
    
    try:
        df = pd.read_csv(db_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading nutrition database: {str(e)}")


def search_food(nutrition_db, food_name, fuzzy=True):
    """
    Search for food in nutrition database
    
    Args:
        nutrition_db (pd.DataFrame): Nutrition database
        food_name (str): Name of food to search
        fuzzy (bool): Use fuzzy matching (case-insensitive, partial match)
    
    Returns:
        pd.DataFrame: Matching rows from database
    """
    if nutrition_db.empty:
        return pd.DataFrame()
    
    if fuzzy:
        # Case-insensitive partial match
        mask = nutrition_db['food_name'].str.lower().str.contains(
            food_name.lower(), na=False
        )
        matches = nutrition_db[mask]
    else:
        # Exact match
        matches = nutrition_db[
            nutrition_db['food_name'].str.lower() == food_name.lower()
        ]
    
    return matches


def get_nutrition(food_name, db_path="data/nutrition_db.csv", portion_size=100):
    """
    Get nutritional information for a food item
    
    Args:
        food_name (str): Name of the food
        db_path (str): Path to nutrition database
        portion_size (float): Portion size in grams (default: 100g)
    
    Returns:
        dict: Nutritional information, or None if not found
    """
    # Load database
    nutrition_db = load_nutrition_database(db_path)
    
    if nutrition_db.empty:
        return None
    
    # Search for food
    matches = search_food(nutrition_db, food_name)
    
    if matches.empty:
        return None
    
    # Get first match
    food_data = matches.iloc[0]
    
    # Calculate nutrition based on portion size
    scale_factor = portion_size / food_data['per_100g']
    
    nutrition = {
        'food_name': food_data['food_name'],
        'portion_size_g': portion_size,
        'calories': round(food_data['calories'] * scale_factor, 1),
        'fat_g': round(food_data['fat_g'] * scale_factor, 2),
        'carbs_g': round(food_data['carbs_g'] * scale_factor, 2),
        'protein_g': round(food_data['protein_g'] * scale_factor, 2),
        'fiber_g': round(food_data['fiber_g'] * scale_factor, 2),
    }
    
    return nutrition


def create_sample_nutrition_db(output_path="data/nutrition_db.csv"):
    """
    Create a sample nutrition database with common Indian foods
    This is a placeholder - you should replace with real data
    
    Args:
        output_path (str): Path to save the CSV file
    """
    # Sample Indian food nutrition data (per 100g)
    sample_data = {
        'food_name': [
            'Biryani', 'Dosa', 'Idli', 'Samosa', 'Curry',
            'Naan', 'Roti', 'Dal', 'Paneer Tikka', 'Butter Chicken',
            'Palak Paneer', 'Chole', 'Rajma', 'Aloo Gobi', 'Baingan Bharta'
        ],
        'calories': [
            350, 150, 100, 260, 200,
            310, 300, 120, 280, 250,
            180, 200, 150, 120, 100
        ],
        'fat_g': [
            12.5, 5.0, 2.0, 15.0, 10.0,
            8.0, 7.0, 3.0, 18.0, 15.0,
            12.0, 8.0, 5.0, 4.0, 6.0
        ],
        'carbs_g': [
            45.0, 25.0, 18.0, 30.0, 15.0,
            50.0, 55.0, 20.0, 10.0, 8.0,
            8.0, 30.0, 25.0, 20.0, 12.0
        ],
        'protein_g': [
            15.0, 4.0, 3.0, 8.0, 12.0,
            10.0, 9.0, 7.0, 20.0, 18.0,
            10.0, 8.0, 7.0, 4.0, 3.0
        ],
        'fiber_g': [
            3.0, 2.0, 1.5, 2.5, 3.0,
            2.0, 2.5, 5.0, 1.0, 1.5,
            2.0, 6.0, 8.0, 4.0, 3.0
        ],
        'per_100g': [
            100, 100, 100, 100, 100,
            100, 100, 100, 100, 100,
            100, 100, 100, 100, 100
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample nutrition database created at {output_path}")
    
    return df

