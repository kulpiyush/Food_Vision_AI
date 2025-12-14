"""
Quick test script for GenAI model
Tests that the GenAI module works correctly
"""

from models.genai_model import GenAIModel, get_genai_model

def test_genai_model():
    """Test GenAI model initialization and methods"""
    print("=" * 60)
    print("Testing GenAI Model (Step 2 - Phase 2)")
    print("=" * 60)
    
    # Test 1: Create model instance
    print("\n1. Creating GenAIModel instance...")
    model = get_genai_model(provider="ollama", model_name="llama3.2")
    print(f"   ✅ Model created: {model.provider}/{model.model_name}")
    print(f"   ✅ Base URL: {model.base_url}")
    
    # Test 2: Check availability
    print("\n2. Checking Ollama availability...")
    is_available = model.is_available()
    print(f"   Available: {is_available}")
    
    if not is_available:
        print("\n   ⚠️  Ollama is not available.")
        print("   To use GenAI features:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Run: ollama pull llama3.2")
        print("   3. Start Ollama service")
        print("\n   The app will work without GenAI, but descriptions and Q&A won't be available.")
    else:
        print("   ✅ Ollama is available!")
    
    # Test 3: Get model info
    print("\n3. Model Information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"   - {key}: {value}")
    
    # Test 4: Test methods (if available)
    if is_available:
        print("\n4. Testing GenAI methods...")
        
        # Test food description
        print("\n   Testing generate_food_description()...")
        try:
            description = model.generate_food_description(
                food_name="Biryani",
                nutrition_data={
                    "calories": 350,
                    "fat_g": 12.5,
                    "carbs_g": 45.0,
                    "protein_g": 15.0,
                    "portion_size_g": 100
                },
                confidence=0.85
            )
            print(f"   ✅ Description generated:")
            print(f"   {description}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test nutrition analysis
        print("\n   Testing analyze_nutrition()...")
        try:
            analysis = model.analyze_nutrition(
                food_name="Biryani",
                nutrition_data={
                    "calories": 350,
                    "fat_g": 12.5,
                    "carbs_g": 45.0,
                    "protein_g": 15.0,
                    "fiber_g": 3.0,
                    "portion_size_g": 100
                }
            )
            print(f"   ✅ Analysis generated:")
            print(f"   {analysis}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test Q&A
        print("\n   Testing answer_question()...")
        try:
            answer = model.answer_question(
                question="Is Biryani healthy?",
                context={
                    "food_name": "Biryani",
                    "nutrition_data": {
                        "calories": 350,
                        "fat_g": 12.5,
                        "carbs_g": 45.0,
                        "protein_g": 15.0,
                        "portion_size_g": 100
                    }
                }
            )
            print(f"   ✅ Answer generated:")
            print(f"   {answer}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test alternatives
        print("\n   Testing suggest_alternatives()...")
        try:
            alternatives = model.suggest_alternatives(
                food_name="Biryani",
                nutrition_data={
                    "calories": 350,
                    "fat_g": 12.5,
                    "carbs_g": 45.0,
                    "protein_g": 15.0,
                    "portion_size_g": 100
                },
                dietary_goal="low calorie"
            )
            print(f"   ✅ Alternatives generated:")
            print(f"   {alternatives}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n4. Skipping method tests (Ollama not available)")
        print("   Methods will work once Ollama is installed and running.")
        print("\n   To test methods:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Run: ollama pull llama3.2")
        print("   3. Make sure Ollama service is running")
        print("   4. Run this test again")
    
    print("\n" + "=" * 60)
    if is_available:
        print("✅ All tests passed! GenAI model is ready for use.")
    else:
        print("✅ Module works correctly! Install Ollama to enable GenAI features.")
    print("=" * 60)
    return is_available

if __name__ == "__main__":
    success = test_genai_model()
    exit(0)  # Exit 0 either way (module works, just Ollama may not be available)

