"""
Generative AI Model Wrapper
Handles integration with Ollama (Llama, Mistral, etc.) for food descriptions and Q&A
Phase 2: GenAI integration for descriptions and interactive Q&A
"""

import requests
import json
from typing import Dict, List, Optional, Any
import warnings
import os


class GenAIModel:
    """
    Wrapper class for Generative AI models (Ollama, OpenAI, Anthropic)
    Phase 2: Primary support for Ollama with Llama 3.2
    """
    
    def __init__(
        self,
        provider="ollama",
        model_name="llama3.2",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=500,
        api_key=None
    ):
        """
        Initialize GenAI model
        
        Args:
            provider (str): AI provider ("ollama", "openai", "anthropic")
            model_name (str): Model name (e.g., "llama3.2", "mistral", "gpt-4o-mini")
            base_url (str): Base URL for API (default: Ollama localhost)
            temperature (float): Sampling temperature (0.0-1.0)
            max_tokens (int): Maximum tokens in response
            api_key (str): API key (for OpenAI/Anthropic, optional for Ollama)
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self._is_available = None
        
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available and running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            return False
    
    def is_available(self) -> bool:
        """Check if the GenAI service is available"""
        if self._is_available is not None:
            return self._is_available
        
        if self.provider == "ollama":
            self._is_available = self._check_ollama_available()
        elif self.provider in ["openai", "anthropic"]:
            # For API providers, assume available if API key is set
            self._is_available = self.api_key is not None
        else:
            self._is_available = False
        
        return self._is_available
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Ollama API"""
        if not self.is_available():
            raise RuntimeError("Ollama is not available. Please install and start Ollama: https://ollama.ai")
        
        url = f"{self.base_url}/api/generate"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Combine system prompt and user prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Ollama: {str(e)}")
    
    def _call_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI API (future implementation)"""
        # Placeholder for OpenAI integration
        raise NotImplementedError("OpenAI integration coming soon. Use Ollama for now.")
    
    def _call_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Anthropic API (future implementation)"""
        # Placeholder for Anthropic integration
        raise NotImplementedError("Anthropic integration coming soon. Use Ollama for now.")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using the configured GenAI model
        
        Args:
            prompt (str): User prompt
            system_prompt (str): System prompt (optional)
        
        Returns:
            str: Generated text
        """
        if not self.is_available():
            raise RuntimeError(
                f"{self.provider} is not available. "
                f"For Ollama: Install from https://ollama.ai and run 'ollama pull {self.model_name}'"
            )
        
        if self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_food_description(
        self,
        food_name: str,
        nutrition_data: Optional[Dict] = None,
        confidence: Optional[float] = None
    ) -> str:
        """
        Generate a description of the detected food
        
        Args:
            food_name (str): Name of the detected food
            nutrition_data (dict): Nutritional information (optional)
            confidence (float): Detection confidence (optional)
        
        Returns:
            str: Food description
        """
        system_prompt = """You are a helpful assistant that describes Indian food dishes. 
Provide informative, appetizing descriptions that highlight the dish's characteristics, 
ingredients, and cultural significance. Keep descriptions concise (2-3 sentences)."""
        
        prompt = f"Describe the Indian dish '{food_name}'."
        
        if nutrition_data:
            prompt += f"\n\nNutritional information (per {nutrition_data.get('portion_size_g', 100)}g):"
            prompt += f"\n- Calories: {nutrition_data.get('calories', 'N/A')}"
            prompt += f"\n- Fat: {nutrition_data.get('fat_g', 'N/A')}g"
            prompt += f"\n- Carbs: {nutrition_data.get('carbs_g', 'N/A')}g"
            prompt += f"\n- Protein: {nutrition_data.get('protein_g', 'N/A')}g"
        
        if confidence:
            prompt += f"\n\nNote: This dish was detected with {confidence*100:.1f}% confidence."
        
        try:
            return self.generate(prompt, system_prompt)
        except Exception as e:
            return f"Unable to generate description: {str(e)}"
    
    def analyze_nutrition(
        self,
        food_name: str,
        nutrition_data: Dict
    ) -> str:
        """
        Analyze nutritional information and provide insights
        
        Args:
            food_name (str): Name of the food
            nutrition_data (dict): Nutritional information
        
        Returns:
            str: Nutritional analysis
        """
        system_prompt = """You are a nutrition expert specializing in Indian cuisine. 
Provide helpful nutritional insights about dishes, including health benefits, 
dietary considerations, and recommendations. Keep responses concise and informative."""
        
        prompt = f"Analyze the nutritional value of '{food_name}':\n\n"
        prompt += f"Per {nutrition_data.get('portion_size_g', 100)}g serving:\n"
        prompt += f"- Calories: {nutrition_data.get('calories', 'N/A')}\n"
        prompt += f"- Fat: {nutrition_data.get('fat_g', 'N/A')}g\n"
        prompt += f"- Carbohydrates: {nutrition_data.get('carbs_g', 'N/A')}g\n"
        prompt += f"- Protein: {nutrition_data.get('protein_g', 'N/A')}g\n"
        prompt += f"- Fiber: {nutrition_data.get('fiber_g', 'N/A')}g\n\n"
        prompt += "Provide a brief nutritional analysis and health insights."
        
        try:
            return self.generate(prompt, system_prompt)
        except Exception as e:
            return f"Unable to analyze nutrition: {str(e)}"
    
    def answer_question(
        self,
        question: str,
        context: Dict
    ) -> str:
        """
        Answer a question about the detected food
        
        Args:
            question (str): User's question
            context (dict): Context including food_name, nutrition_data, etc.
        
        Returns:
            str: Answer to the question
        """
        system_prompt = """You are a helpful assistant knowledgeable about Indian cuisine, 
nutrition, and cooking. Answer questions accurately and helpfully based on the provided context. 
If you don't know something, say so politely."""
        
        food_name = context.get("food_name", "the dish")
        nutrition_data = context.get("nutrition_data")
        
        prompt = f"Context: The user is asking about '{food_name}'.\n\n"
        
        if nutrition_data:
            prompt += f"Nutritional information:\n"
            prompt += f"- Calories: {nutrition_data.get('calories', 'N/A')} per {nutrition_data.get('portion_size_g', 100)}g\n"
            prompt += f"- Fat: {nutrition_data.get('fat_g', 'N/A')}g\n"
            prompt += f"- Carbs: {nutrition_data.get('carbs_g', 'N/A')}g\n"
            prompt += f"- Protein: {nutrition_data.get('protein_g', 'N/A')}g\n\n"
        
        prompt += f"Question: {question}\n\n"
        prompt += "Please provide a helpful answer."
        
        try:
            return self.generate(prompt, system_prompt)
        except Exception as e:
            return f"Unable to answer question: {str(e)}"
    
    def suggest_alternatives(
        self,
        food_name: str,
        nutrition_data: Dict,
        dietary_goal: Optional[str] = None
    ) -> str:
        """
        Suggest healthier alternatives or modifications
        
        Args:
            food_name (str): Name of the food
            nutrition_data (dict): Nutritional information
            dietary_goal (str): Dietary goal (e.g., "low calorie", "high protein")
        
        Returns:
            str: Suggestions for alternatives or modifications
        """
        system_prompt = """You are a nutrition expert. Suggest healthier alternatives 
or modifications to Indian dishes based on dietary goals. Be practical and specific."""
        
        prompt = f"Suggest healthier alternatives or modifications for '{food_name}'"
        
        if dietary_goal:
            prompt += f" with a focus on: {dietary_goal}"
        
        prompt += f"\n\nCurrent nutrition (per {nutrition_data.get('portion_size_g', 100)}g):"
        prompt += f"\n- Calories: {nutrition_data.get('calories', 'N/A')}"
        prompt += f"\n- Fat: {nutrition_data.get('fat_g', 'N/A')}g"
        prompt += f"\n- Carbs: {nutrition_data.get('carbs_g', 'N/A')}g"
        prompt += f"\n- Protein: {nutrition_data.get('protein_g', 'N/A')}g"
        
        try:
            return self.generate(prompt, system_prompt)
        except Exception as e:
            return f"Unable to generate suggestions: {str(e)}"
    
    def get_model_info(self) -> Dict:
        """Get information about the GenAI model"""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "is_available": self.is_available(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


def get_genai_model(
    provider="ollama",
    model_name="llama3.2",
    base_url="http://localhost:11434"
) -> GenAIModel:
    """
    Factory function to create a GenAI model
    Convenience function for easy model initialization
    
    Args:
        provider (str): AI provider
        model_name (str): Model name
        base_url (str): Base URL for API
    
    Returns:
        GenAIModel: Initialized GenAI model
    """
    return GenAIModel(
        provider=provider,
        model_name=model_name,
        base_url=base_url
    )

