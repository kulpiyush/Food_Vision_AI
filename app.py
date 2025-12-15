"""
FoodVisionAI - Main Streamlit Application
Phase 2: Real vision model + GenAI integration
"""

import streamlit as st
import pandas as pd
from PIL import Image
import os
from pathlib import Path

# Import utility functions
from utils.nutrition_calculator import get_nutrition
from models.vision_model import get_vision_model, VisionModel
from models.genai_model import get_genai_model, GenAIModel


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

# Page configuration
st.set_page_config(
    page_title="FoodVisionAI - Nutritional Analysis",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'vision_model' not in st.session_state:
    st.session_state.vision_model = None
if 'genai_model' not in st.session_state:
    st.session_state.genai_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'food_description' not in st.session_state:
    st.session_state.food_description = None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🍽️ FoodVisionAI - Nutritional Analysis</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Food Recognition and Nutritional Analysis for Indian Cuisine")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("---")
        
        # Model selection
        st.subheader("Model Settings")
        model_choice = st.selectbox(
            "Vision Model",
            ["EfficientNet-B0 (Recommended)", "ResNet-50", "MobileNet-V2"],
            help="Classification models for single-dish recognition"
        )
        
        st.markdown("---")
        
        # Info section
        st.subheader("ℹ️ About")
        st.info("""
        **FoodVisionAI** analyzes food images to provide:
        - Food identification
        - Nutritional information
        - AI-generated descriptions
        - Meal recommendations
        """)
        
        st.markdown("---")
        
        # Model status
        if st.session_state.vision_model is not None:
            model_info = st.session_state.vision_model.get_model_info()
            model_type = "Fine-tuned" if model_info.get('is_fine_tuned', False) else "Pretrained"
            st.success(f"✅ Vision Model: {model_type} {model_info['model_name']} loaded")
            st.caption(f"Device: {model_info['device']} | Classes: {model_info['num_classes']}")
        else:
            st.info("ℹ️ Vision model will load on first analysis")
        
        st.markdown("---")
        
        # GenAI status
        if st.session_state.genai_model is not None:
            genai_info = st.session_state.genai_model.get_model_info()
            if genai_info['is_available']:
                st.success(f"✅ GenAI: {genai_info['model_name']} available")
            else:
                st.warning(f"⚠️ GenAI: {genai_info['model_name']} not available")
                st.caption("Install Ollama to enable GenAI features")
        else:
            st.info("ℹ️ GenAI will initialize on first use")
        
        st.markdown("---")
        st.caption("**Phase 2:** Real vision model + GenAI integration")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Food Image</div>', unsafe_allow_html=True)
        
        # Image upload widget
        uploaded_file = st.file_uploader(
            "Choose an image of Indian food...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of Indian cuisine (Biryani, Dosa, Curry, etc.)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Store in session state
            st.session_state.uploaded_image = image
            
            # Image info
            st.success(f"✅ Image uploaded successfully!")
            st.info(f"**Image Details:** {image.size[0]}x{image.size[1]} pixels, Format: {image.format}")
            
            # Analyze button
            st.markdown("---")
            analyze_button = st.button("🔍 Analyze Food", type="primary", use_container_width=True)
            
            if analyze_button:
                with st.spinner("Analyzing food image..."):
                    # Validate image
                    is_valid, error_msg = validate_image(image)
                    if not is_valid:
                        st.error(f"❌ {error_msg}")
                    else:
                        try:
                            # Load model if not already loaded (cache in session state)
                            if st.session_state.vision_model is None:
                                with st.spinner("Loading classification model (first time only)..."):
                                    # Determine model name from selection
                                    model_name_map = {
                                        "EfficientNet-B0 (Recommended)": "efficientnet_b0",
                                        "ResNet-50": "resnet50",
                                        "MobileNet-V2": "mobilenet_v2"
                                    }
                                    selected_model = model_name_map.get(model_choice, "efficientnet_b0")
                                    
                                    # Try to use fine-tuned model if available, otherwise use pretrained
                                    model_path = "models/weights/food_classifier.pt"
                                    
                                    # Load class names from saved model weights directory
                                    class_names_file = "models/weights/class_names.txt"
                                    class_names = None
                                    if os.path.exists(class_names_file):
                                        with open(class_names_file, 'r') as f:
                                            class_names = [line.strip() for line in f if line.strip()]
                                        st.info(f"✅ Loaded {len(class_names)} dish classes from trained model")
                                    
                                    if os.path.exists(model_path):
                                        st.session_state.vision_model = get_vision_model(
                                            model_name=selected_model,
                                            model_path=model_path,
                                            class_names=class_names
                                        )
                                        st.success("✅ Loaded fine-tuned model (93.25% accuracy)")
                                    else:
                                        # Use pretrained model (will need fine-tuning for food detection)
                                        st.session_state.vision_model = get_vision_model(
                                            model_name=selected_model,
                                            class_names=class_names
                                        )
                                        st.warning("⚠️ Using pretrained model. Train a model first for better accuracy.")
                            
                            # Make prediction with classification model
                            prediction = st.session_state.vision_model.predict(image)
                            
                            # Get nutrition data from Kaggle Indian Food Nutrition Dataset (direct semantic matching)
                            nutrition = get_nutrition(
                                prediction["food_name"],
                                kaggle_path="data/Indian_Food_Nutrition_Processed.csv",
                                portion_size=100,
                                similarity_threshold=0.5
                            )
                            
                            # Initialize GenAI model if not already loaded
                            if st.session_state.genai_model is None:
                                st.session_state.genai_model = get_genai_model(
                                    provider="ollama",
                                    model_name="llama3.2"
                                )
                            
                            # Generate food description if GenAI is available
                            food_description = None
                            if st.session_state.genai_model.is_available():
                                try:
                                    with st.spinner("🤖 Generating AI description..."):
                                        food_description = st.session_state.genai_model.generate_food_description(
                                            food_name=prediction["food_name"],
                                            nutrition_data=nutrition,
                                            confidence=prediction.get("confidence", 0.0)
                                        )
                                        st.session_state.food_description = food_description
                                except Exception as e:
                                    st.warning(f"⚠️ Could not generate description: {str(e)}")
                                    st.session_state.food_description = None
                            else:
                                st.session_state.food_description = None
                            
                            # Clear chat history for new analysis
                            st.session_state.chat_history = []
                            
                            # Store results
                            st.session_state.analysis_results = {
                                "prediction": prediction,
                                "nutrition": nutrition,
                                "status": prediction.get("status", "pretrained"),
                                "food_description": food_description
                            }
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.exception(e)
                            st.info("💡 Tip: Make sure the image is clear and shows food clearly")
        else:
            st.info("👆 Please upload an image to get started")
            # Show example placeholder
            st.markdown("**Example:** Upload an image of Biryani, Dosa, Curry, or any Indian dish")
    
    with col2:
        st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results is not None:
            # Display results
            results = st.session_state.analysis_results
            prediction = results.get("prediction", {})
            nutrition = results.get("nutrition")
            
            # Display results
            st.markdown("---")
            st.subheader("🔍 Detection Results")
            
            # Check for out-of-distribution (OOD) detection FIRST - before showing any results
            is_ood = prediction.get("is_out_of_distribution", False)
            ood_reason = prediction.get("ood_reason", "")
            
            if is_ood:
                # Don't show incorrect detection results for OOD inputs
                st.error("⚠️ **Out-of-Distribution Detection**")
                st.warning(
                    "**This app is trained specifically on Indian dishes and works best with Indian cuisine images.**\n\n"
                    "The uploaded image does not appear to be an Indian dish. Please upload an image of an Indian dish "
                    "(e.g., Biryani, Dosa, Curry, Samosa, etc.) for accurate food recognition and nutrition analysis."
                )
                if ood_reason:
                    st.caption(f"🔍 Detection reason: {ood_reason}")
                
                # Show that no food was detected
                st.info("❌ **No Indian dish detected** - Unable to provide nutrition information for non-Indian foods.")
                
                # Skip showing incorrect detection results
                st.markdown("---")
                st.subheader("📊 Nutritional Information")
                st.warning("⚠️ Nutrition data unavailable - Please upload an image of an Indian dish.")
                
                st.markdown("---")
                st.subheader("📝 AI Description")
                st.info("💡 Upload an image of an Indian dish to get AI-generated descriptions and nutritional information.")
                
                # Stop here - don't show incorrect results
                return
            
            # If not OOD, show normal detection results
            # Food detection
            food_name = prediction.get("food_name", "Unknown")
            confidence = prediction.get("confidence", 0.0)
            status = prediction.get("status", "unknown")
            model_name = prediction.get("model_name", "efficientnet_b0")
            
            st.success(f"**Detected Food:** {food_name}")
            
            # Check if detection is uncertain
            is_uncertain = prediction.get("is_uncertain", False)
            if is_uncertain:
                st.warning("⚠️ **Low Confidence Detection** - The model is not very confident about this detection. Results may be inaccurate.")
            
            # Confidence metric with color coding
            confidence_pct = confidence * 100
            if confidence_pct >= 65:
                st.metric("Primary Confidence", f"{confidence_pct:.1f}%", delta="High confidence")
            elif confidence_pct >= 50:
                st.metric("Primary Confidence", f"{confidence_pct:.1f}%", delta="Medium confidence", delta_color="off")
            else:
                st.metric("Primary Confidence", f"{confidence_pct:.1f}%", delta="Low confidence", delta_color="inverse")
                st.warning("⚠️ **Very Low Confidence** - This detection may be incorrect.")
            
            # Model status info
            if status == "pretrained":
                st.info("ℹ️ Using pretrained ImageNet model. Fine-tune on Khana dataset for better accuracy.")
            elif status == "fine_tuned":
                st.success("✅ Using fine-tuned model optimized for Indian cuisine")
                st.caption(f"🎯 Validation Accuracy: 93.25% | Trained on 105K+ images | 80 dish classes")
            
            # Top predictions
            if "top_predictions" in prediction and len(prediction["top_predictions"]) > 0:
                st.markdown("**Top Predictions:**")
                for i, pred in enumerate(prediction["top_predictions"][:3], 1):
                    conf_pct = pred['confidence'] * 100
                    st.caption(f"{i}. **{pred['food_name']}** ({conf_pct:.1f}%)")
            
            st.markdown("---")
            st.subheader("📊 Nutritional Information")
            
            if nutrition:
                # Display nutrition data
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Calories", f"{nutrition['calories']:.0f}")
                with col2:
                    st.metric("Fat", f"{nutrition['fat_g']:.1f}g")
                with col3:
                    st.metric("Carbs", f"{nutrition['carbs_g']:.1f}g")
                with col4:
                    st.metric("Protein", f"{nutrition['protein_g']:.1f}g")
                
                st.caption(f"Per {nutrition['portion_size_g']}g serving")
            else:
                st.warning(f"⚠️ Nutrition data not found for '{food_name}'")
                st.info("💡 Add this food to the nutrition database to see nutritional information")
            
            st.markdown("---")
            st.subheader("📝 AI Description")
            
            # Show food description
            if st.session_state.food_description:
                st.markdown(st.session_state.food_description)
                st.caption("🤖 Generated by AI")
            elif st.session_state.genai_model and st.session_state.genai_model.is_available():
                st.info("💡 Click 'Analyze Food' again to generate description")
            else:
                st.warning("⚠️ GenAI not available. Install Ollama to enable AI descriptions.")
                with st.expander("How to enable GenAI"):
                    st.markdown("""
                    1. Install Ollama: https://ollama.ai
                    2. Run: `ollama pull llama3.2`
                    3. Restart this app
                    
                    See `OLLAMA_SETUP.md` for detailed instructions.
                    """)
            
            st.markdown("---")
            st.subheader("💬 Q&A Interface")
            
            # Q&A Chat Interface
            if st.session_state.genai_model and st.session_state.genai_model.is_available():
                # Display chat history
                if st.session_state.chat_history:
                    st.markdown("**Chat History:**")
                    for i, chat in enumerate(st.session_state.chat_history):
                        with st.chat_message("user"):
                            st.write(chat["question"])
                        with st.chat_message("assistant"):
                            st.write(chat["answer"])
                
                # Question input
                question = st.text_input(
                    "Ask a question about this food:",
                    placeholder="e.g., Is this healthy? What are the health benefits?",
                    key="qa_input"
                )
                
                if st.button("💬 Ask", type="secondary"):
                    if question:
                        try:
                            with st.spinner("🤖 Thinking..."):
                                context = {
                                    "food_name": food_name,
                                    "nutrition_data": nutrition,
                                    "confidence": confidence
                                }
                                answer = st.session_state.genai_model.answer_question(
                                    question=question,
                                    context=context
                                )
                                
                                # Add to chat history
                                st.session_state.chat_history.append({
                                    "question": question,
                                    "answer": answer
                                })
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.caption("💡 Try asking: 'Is this healthy?', 'What are the health benefits?', 'How can I make this healthier?'")
            else:
                st.warning("⚠️ GenAI not available. Install Ollama to enable Q&A.")
                with st.expander("How to enable Q&A"):
                    st.markdown("""
                    1. Install Ollama: https://ollama.ai
                    2. Run: `ollama pull llama3.2`
                    3. Restart this app
                    """)
            
            # Phase 2 info
            st.markdown("---")
            genai_status = "✅ Active" if (st.session_state.genai_model and st.session_state.genai_model.is_available()) else "⚠️ Not available"
            st.caption(f"ℹ️ **Status:** Classification model ({model_name}) + GenAI ({genai_status})")
        else:
            st.info("👈 Upload an image and click 'Analyze Food' to see results here")
            
            # Instructions
            st.markdown("---")
            st.markdown("### 📋 How to Use:")
            st.markdown("""
            1. **Upload** an image of Indian food
            2. **Click** the "Analyze Food" button
            3. **View** the analysis results
            4. **Ask questions** about the meal (Phase 2)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>FoodVisionAI - Automated Nutritional Analysis App</p>
        <p><small>Phase 2: Real Vision Model + GenAI | Built with Streamlit, PyTorch & Ollama</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

