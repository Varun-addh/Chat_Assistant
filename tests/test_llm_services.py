"""Test script to verify LLM service configuration"""
import os

import pytest

# This file is a manual/integration diagnostic script. It depends on external
# LLM credentials and network access. Skip during normal pytest runs unless
# explicitly enabled.
if __name__ != "__main__" and os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "llm integration test (set RUN_INTEGRATION_TESTS=1 to run)",
        allow_module_level=True,
    )

import asyncio
from app.services.llm_service import get_llm_service
from app.config import settings

async def test_llm_services():
    print("=" * 60)
    print("Testing LLM Service Configuration")
    print("=" * 60)
    
    # Test global settings
    print(f"\n📋 Global Settings:")
    print(f"  LLM_PROVIDER: {settings.llm_provider}")
    print(f"  GROQ_API_KEY: {'✓ Set' if settings.groq_api_key else '✗ Not Set'}")
    print(f"  GEMINI_API_KEY: {'✓ Set' if settings.gemini_api_key else '✗ Not Set'}")
    print(f"  GROQ_MODEL: {settings.groq_model}")
    print(f"  GEMINI_MODEL: {settings.gemini_model}")
    
    # Test Gemini service (for answer card)
    print(f"\n🔷 Gemini Service (Answer Card):")
    gemini_service = get_llm_service("gemini")
    print(f"  Instance Settings Provider: {gemini_service._settings.llm_provider}")
    print(f"  API Key Set: {'✓' if gemini_service._settings.gemini_api_key else '✗'}")
    print(f"  Enabled: {'✓ YES' if gemini_service.enabled else '✗ NO'}")
    print(f"  Client: {type(gemini_service._ensure_client())}")
    
    # Test Groq service (for Interview Intelligence & Mock Interview)
    print(f"\n🔶 Groq Service (Interview Intelligence & Mock Interview):")
    groq_service = get_llm_service("groq")
    print(f"  Instance Settings Provider: {groq_service._settings.llm_provider}")
    print(f"  API Key Set: {'✓' if groq_service._settings.groq_api_key else '✗'}")
    print(f"  Enabled: {'✓ YES' if groq_service.enabled else '✗ NO'}")
    print(f"  Client: {type(groq_service._ensure_client())}")
    
    # Test actual LLM call with Groq
    print(f"\n🧪 Testing Groq LLM Call:")
    try:
        test_prompt = "Say 'Hello from Groq!' and nothing else."
        response = await groq_service.generate_answer(test_prompt)
        print(f"  Response: {response[:100]}...")
        print(f"  ✓ SUCCESS - Groq LLM is working!")
    except Exception as e:
        print(f"  ✗ FAILED - {type(e).__name__}: {e}")
    
    # Test actual LLM call with Gemini
    print(f"\n🧪 Testing Gemini LLM Call:")
    try:
        test_prompt = "Say 'Hello from Gemini!' and nothing else."
        response = await gemini_service.generate_answer(test_prompt)
        print(f"  Response: {response[:100]}...")
        print(f"  ✓ SUCCESS - Gemini LLM is working!")
    except Exception as e:
        print(f"  ✗ FAILED - {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_llm_services())
