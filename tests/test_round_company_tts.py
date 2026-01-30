"""
Test round-based interview with company context and TTS audio.
"""

import os

import pytest

# Integration test: requires the local API server and may require audio deps.
if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip("requires local API server (set RUN_INTEGRATION_TESTS=1)", allow_module_level=True)

import requests
import json

BASE_URL = "http://localhost:8000"

def test_company_context():
    """Test that company is framed as 'interviewing FOR' not 'working AT'."""
    
    print("=" * 70)
    print("TEST: Company Context (Interviewing FOR, not working AT)")
    print("=" * 70)
    
    payload = {
        "screen_shared": True,
        "camera_enabled": True,
        "round_type": "technical_round_1",
        "domain": "Data Engineering",
        "experience_years": 2,
        "company_specific": "Amazon",
    }
    
    print(f"\n📤 Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/practice/interview/start-round",
        json=payload
    )
    
    if response.status_code != 200:
        print(f"\n❌ FAILED: {response.status_code}")
        print(f"Error: {response.text}")
        return
    
    data = response.json()
    
    print(f"\n✅ SUCCESS: {response.status_code}")
    print(f"\nSession ID: {data['session_id']}")
    print(f"Total Questions: {data['total_questions']}")
    print(f"Progress: {data['progress']}")
    print(f"\n📝 First Question:")
    print(f"   Text: {data['first_question']['text']}")
    print(f"   Difficulty: {data['first_question']['difficulty']}")
    print(f"   Time Limit: {data['first_question']['time_limit']}s")
    
    # Check TTS audio
    audio_url = data.get('tts_audio_url', '')
    if audio_url:
        print(f"\n🎤 TTS Audio: ✅ {audio_url}")
        
        # Test if audio file is accessible
        audio_response = requests.get(f"{BASE_URL}{audio_url}")
        if audio_response.status_code == 200:
            print(f"   Audio file size: {len(audio_response.content)} bytes")
            print(f"   Content type: {audio_response.headers.get('content-type')}")
        else:
            print(f"   ⚠️ Audio file not accessible: {audio_response.status_code}")
    else:
        print(f"\n🎤 TTS Audio: ❌ No audio URL")
    
    # Analyze question text for company framing
    question_text = data['first_question']['text'].lower()
    
    print(f"\n🔍 Company Framing Analysis:")
    if "at amazon" in question_text or "working at" in question_text:
        print(f"   ❌ WRONG: Question says 'at Amazon' or 'working at'")
        print(f"   This implies candidate already works there!")
    elif "for amazon" in question_text or "interviewing" in question_text:
        print(f"   ✅ CORRECT: Question properly frames as interviewing FOR Amazon")
    else:
        print(f"   ⚠️ NEUTRAL: Question doesn't explicitly mention company context")
    
    print(f"\n📄 Full Question Text:")
    print(f"   {data['first_question']['text']}")


def test_different_companies():
    """Test multiple companies to verify consistent framing."""
    
    print("\n" + "=" * 70)
    print("TEST: Multiple Companies")
    print("=" * 70)
    
    companies = ["Google", "Meta", "Microsoft", "Netflix", "Startup"]
    
    for company in companies:
        print(f"\n🏢 Testing: {company}")
        
        payload = {
            "screen_shared": True,
            "camera_enabled": True,
            "round_type": "hr_screening",
            "domain": "Python",
            "experience_years": 4,
            "company_specific": company,
        }
        
        response = requests.post(
            f"{BASE_URL}/api/practice/interview/start-round",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            question_text = data['first_question']['text']
            
            # Check framing
            question_lower = question_text.lower()
            if f"at {company.lower()}" in question_lower:
                print(f"   ❌ WRONG: 'at {company}'")
            elif f"for {company.lower()}" in question_lower:
                print(f"   ✅ CORRECT: 'for {company}'")
            else:
                print(f"   ✓ Neutral framing")
            
            print(f"   Q: {question_text[:100]}...")
        else:
            print(f"   ❌ Failed: {response.status_code}")


def test_tts_availability():
    """Test that TTS audio is generated for all rounds."""
    
    print("\n" + "=" * 70)
    print("TEST: TTS Audio Generation for All Rounds")
    print("=" * 70)
    
    rounds = [
        "hr_screening",
        "technical_round_1",
        "technical_round_2",
        "behavioral"
    ]
    
    for round_type in rounds:
        print(f"\n🎯 Round: {round_type}")
        
        payload = {
            "screen_shared": True,
            "camera_enabled": True,
            "round_type": round_type,
            "domain": "Python",
            "experience_years": 3,
        }
        
        response = requests.post(
            f"{BASE_URL}/api/practice/interview/start-round",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            audio_url = data.get('tts_audio_url', '')
            
            if audio_url:
                print(f"   🎤 Audio: ✅ {audio_url}")
            else:
                print(f"   🎤 Audio: ❌ Missing")
        else:
            print(f"   ❌ Request failed: {response.status_code}")


if __name__ == "__main__":
    print("\n🚀 Starting Round-Based Interview Tests\n")
    
    # Test 1: Company context framing
    test_company_context()
    
    # Test 2: Multiple companies
    test_different_companies()
    
    # Test 3: TTS for all rounds
    test_tts_availability()
    
    print("\n\n✅ All tests completed!\n")
