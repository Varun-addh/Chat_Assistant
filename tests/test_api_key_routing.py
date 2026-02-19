"""
Test script to verify API key routing logic.

This demonstrates how the application routes requests to Groq/Gemini based on available API keys:

1. Only Groq API key → All features use Groq
2. Only Gemini API key → All features use Gemini  
3. Both API keys → AI Copilot Chat uses Gemini, other features use Groq
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_provider_routing():
    """Test the provider selection logic"""
    from app.config import Settings
    
    print("=" * 70)
    print("API KEY ROUTING TEST")
    print("=" * 70)
    
    # Test Case 1: Only Groq API key
    print("\n1. ONLY GROQ API KEY CONFIGURED:")
    print("-" * 70)
    settings1 = Settings(groq_api_key="gsk_test123", gemini_api_key=None)
    print(f"   AI Copilot Chat → {settings1.get_effective_provider('copilot').upper()}")
    print(f"   Interview Intelligence → {settings1.get_effective_provider('default').upper()}")
    print(f"   Mock Interview → {settings1.get_effective_provider('default').upper()}")
    print(f"   Other Features → {settings1.get_effective_provider('default').upper()}")
    
    # Test Case 2: Only Gemini API key
    print("\n2. ONLY GEMINI API KEY CONFIGURED:")
    print("-" * 70)
    settings2 = Settings(groq_api_key=None, gemini_api_key="AIzaTest123")
    print(f"   AI Copilot Chat → {settings2.get_effective_provider('copilot').upper()}")
    print(f"   Interview Intelligence → {settings2.get_effective_provider('default').upper()}")
    print(f"   Mock Interview → {settings2.get_effective_provider('default').upper()}")
    print(f"   Other Features → {settings2.get_effective_provider('default').upper()}")
    
    # Test Case 3: Both API keys
    print("\n3. BOTH GROQ AND GEMINI API KEYS CONFIGURED:")
    print("-" * 70)
    settings3 = Settings(groq_api_key="gsk_test123", gemini_api_key="AIzaTest123")
    print(f"   AI Copilot Chat → {settings3.get_effective_provider('copilot').upper()}")
    print(f"   Interview Intelligence → {settings3.get_effective_provider('default').upper()}")
    print(f"   Mock Interview → {settings3.get_effective_provider('default').upper()}")
    print(f"   Other Features → {settings3.get_effective_provider('default').upper()}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE - All routing logic working as expected!")
    print("=" * 70)
    
    # Verify expected behavior
    assert settings1.get_effective_provider('copilot') == 'groq'
    assert settings1.get_effective_provider('default') == 'groq'
    
    assert settings2.get_effective_provider('copilot') == 'gemini'
    assert settings2.get_effective_provider('default') == 'gemini'
    
    assert settings3.get_effective_provider('copilot') == 'gemini'
    assert settings3.get_effective_provider('default') == 'groq'
    
    print("\n✅ All assertions passed!")

if __name__ == "__main__":
    test_provider_routing()
