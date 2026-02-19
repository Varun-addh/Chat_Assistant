"""
Test difficulty level handling in Practice Mode
"""

import os

import pytest

# This is an integration smoke test that requires the API server to be running.
# Skip it by default so `pytest` works in CI and for quick unit validation.
if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip("requires local API server (set RUN_INTEGRATION_TESTS=1)", allow_module_level=True)

import requests
import json

BASE_URL = "http://localhost:8000"

def test_difficulty_levels():
    """Test if difficulty levels are correctly applied"""
    
    print("=" * 80)
    print("TESTING DIFFICULTY LEVELS - Practice Mode")
    print("=" * 80)
    
    test_profile = {
        "domain": "Software Engineer",
        "experience_years": 3,
        "skills": ["Python", "FastAPI", "PostgreSQL"],
        "job_role": "Backend Developer",
        "company_preference": "any",
        "interview_focus": ["technical", "behavioral"]
    }
    
    # Test EASY difficulty
    print("\n" + "=" * 80)
    print("TEST 1: EASY Difficulty")
    print("=" * 80)
    
    easy_request = {
        "screen_shared": True,
        "camera_enabled": True,
        "difficulty": "easy",
        "question_count": 5,
        "user_profile": test_profile
    }
    
    response = requests.post(f"{BASE_URL}/api/practice/interview/start", json=easy_request)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Session created: {data['session_id']}")
        print(f"\nFirst question:")
        print(f"  Text: {data['first_question']['text'][:100]}...")
        print(f"  Difficulty: {data['first_question']['difficulty']}")
        print(f"  Category: {data['first_question']['category']}")
        print(f"  Time Limit: {data['first_question']['time_limit']}s")
        
        # Fetch full session to check all questions
        session_response = requests.get(f"{BASE_URL}/api/practice/session/{data['session_id']}")
        if session_response.status_code == 200:
            session_data = session_response.json()
            print(f"\n📊 All Questions (Expected: mostly EASY):")
            for i, q in enumerate(session_data['questions'], 1):
                print(f"  Q{i}: {q['difficulty']:8} | {q['category']:15} | {q['time_limit']:3}s | {q['text'][:60]}...")
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
    
    # Test MEDIUM difficulty
    print("\n" + "=" * 80)
    print("TEST 2: MEDIUM Difficulty")
    print("=" * 80)
    
    medium_request = {
        "screen_shared": True,
        "camera_enabled": True,
        "difficulty": "medium",
        "question_count": 5,
        "user_profile": test_profile
    }
    
    response = requests.post(f"{BASE_URL}/api/practice/interview/start", json=medium_request)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Session created: {data['session_id']}")
        
        session_response = requests.get(f"{BASE_URL}/api/practice/session/{data['session_id']}")
        if session_response.status_code == 200:
            session_data = session_response.json()
            print(f"\n📊 All Questions (Expected: mix of EASY/MEDIUM/HARD):")
            difficulty_count = {"easy": 0, "medium": 0, "hard": 0}
            for i, q in enumerate(session_data['questions'], 1):
                difficulty_count[q['difficulty']] += 1
                print(f"  Q{i}: {q['difficulty']:8} | {q['category']:15} | {q['time_limit']:3}s | {q['text'][:60]}...")
            
            print(f"\n📈 Difficulty Distribution:")
            print(f"  Easy: {difficulty_count['easy']}/{len(session_data['questions'])}")
            print(f"  Medium: {difficulty_count['medium']}/{len(session_data['questions'])}")
            print(f"  Hard: {difficulty_count['hard']}/{len(session_data['questions'])}")
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
    
    # Test HARD difficulty
    print("\n" + "=" * 80)
    print("TEST 3: HARD Difficulty")
    print("=" * 80)
    
    hard_request = {
        "screen_shared": True,
        "camera_enabled": True,
        "difficulty": "hard",
        "question_count": 5,
        "user_profile": test_profile
    }
    
    response = requests.post(f"{BASE_URL}/api/practice/interview/start", json=hard_request)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Session created: {data['session_id']}")
        
        session_response = requests.get(f"{BASE_URL}/api/practice/session/{data['session_id']}")
        if session_response.status_code == 200:
            session_data = session_response.json()
            print(f"\n📊 All Questions (Expected: mostly MEDIUM/HARD):")
            difficulty_count = {"easy": 0, "medium": 0, "hard": 0}
            for i, q in enumerate(session_data['questions'], 1):
                difficulty_count[q['difficulty']] += 1
                print(f"  Q{i}: {q['difficulty']:8} | {q['category']:15} | {q['time_limit']:3}s | {q['text'][:60]}...")
            
            print(f"\n📈 Difficulty Distribution:")
            print(f"  Easy: {difficulty_count['easy']}/{len(session_data['questions'])}")
            print(f"  Medium: {difficulty_count['medium']}/{len(session_data['questions'])}")
            print(f"  Hard: {difficulty_count['hard']}/{len(session_data['questions'])}")
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)
    print("\n📝 EXPECTED BEHAVIOR:")
    print("  - EASY mode: Should have mostly easy questions (maybe 1-2 medium for progression)")
    print("  - MEDIUM mode: Should have a balanced mix (some easy warm-up, mostly medium, few hard)")
    print("  - HARD mode: Should have mostly medium/hard questions (challenging but fair)")
    print("\nNote: AI decides difficulty per question, so exact distribution may vary!")


if __name__ == "__main__":
    test_difficulty_levels()
