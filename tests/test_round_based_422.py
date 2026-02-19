"""
Test script to debug 422 errors in round-based interview API.
"""

import os

import pytest

# Integration/debug script: requires the local API server.
if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip("requires local API server (set RUN_INTEGRATION_TESTS=1)", allow_module_level=True)

import requests
import json

BASE_URL = "http://localhost:8000"

def test_valid_requests():
    """Test valid round-based interview requests."""
    
    test_cases = [
        {
            "name": "Minimal valid request",
            "payload": {
                "screen_shared": True,
                "camera_enabled": True,
                "round_type": "TECHNICAL_ROUND_1",
                "domain": "Python",
                "experience_years": 3
            }
        },
        {
            "name": "With company specific",
            "payload": {
                "screen_shared": True,
                "camera_enabled": True,
                "round_type": "SYSTEM_DESIGN",
                "domain": "Data Engineering",
                "experience_years": 5,
                "company_specific": "Google"
            }
        },
        {
            "name": "Junior developer",
            "payload": {
                "screen_shared": True,
                "camera_enabled": True,
                "round_type": "HR_SCREENING",
                "domain": "Java",
                "experience_years": 1
            }
        },
        {
            "name": "Senior developer",
            "payload": {
                "screen_shared": True,
                "camera_enabled": True,
                "round_type": "TECHNICAL_ROUND_2",
                "domain": "Python",
                "experience_years": 10
            }
        }
    ]
    
    print("=" * 70)
    print("TESTING VALID REQUESTS")
    print("=" * 70)
    
    for test in test_cases:
        print(f"\n🧪 Test: {test['name']}")
        print(f"📤 Payload: {json.dumps(test['payload'], indent=2)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/practice/interview/start-round",
                json=test['payload'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ SUCCESS: {response.status_code}")
                data = response.json()
                print(f"   Session ID: {data.get('session_id')}")
                print(f"   Total Questions: {data.get('total_questions')}")
            else:
                print(f"❌ FAILED: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")


def test_invalid_requests():
    """Test invalid requests to understand 422 errors."""
    
    invalid_cases = [
        {
            "name": "Missing domain",
            "payload": {
                "round_type": "TECHNICAL_ROUND_1",
                "experience_years": 3
            }
        },
        {
            "name": "Missing round_type",
            "payload": {
                "domain": "Python",
                "experience_years": 3
            }
        },
        {
            "name": "Invalid round_type",
            "payload": {
                "round_type": "INVALID_ROUND",
                "domain": "Python",
                "experience_years": 3
            }
        },
        {
            "name": "Invalid experience_years (negative)",
            "payload": {
                "round_type": "TECHNICAL_ROUND_1",
                "domain": "Python",
                "experience_years": -1
            }
        },
        {
            "name": "Invalid experience_years (too high)",
            "payload": {
                "round_type": "TECHNICAL_ROUND_1",
                "domain": "Python",
                "experience_years": 50
            }
        }
    ]
    
    print("\n" + "=" * 70)
    print("TESTING INVALID REQUESTS (Expected 422 errors)")
    print("=" * 70)
    
    for test in invalid_cases:
        print(f"\n🧪 Test: {test['name']}")
        print(f"📤 Payload: {json.dumps(test['payload'], indent=2)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/practice/interview/start-round",
                json=test['payload'],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📊 Status: {response.status_code}")
            if response.status_code == 422:
                print(f"✅ Expected 422 error received")
                error_detail = response.json().get('detail', [])
                print(f"   Validation errors:")
                for err in error_detail:
                    if isinstance(err, dict):
                        field = err.get('loc', ['unknown'])[-1]
                        msg = err.get('msg', 'Unknown error')
                        print(f"     - {field}: {msg}")
            else:
                print(f"❓ Unexpected status: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")


def test_difficulty_preview():
    """Test the difficulty preview endpoint."""
    
    print("\n" + "=" * 70)
    print("TESTING DIFFICULTY PREVIEW")
    print("=" * 70)
    
    experience_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
    
    for exp in experience_levels:
        try:
            response = requests.get(
                f"{BASE_URL}/api/practice/difficulty-preview",
                params={"experience_years": exp}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"📊 {exp} years → {data['label']} ({data['difficulty']})")
            else:
                print(f"❌ Error for {exp} years: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception for {exp} years: {e}")


def print_frontend_integration_guide():
    """Print the exact request format for frontend."""
    
    print("\n" + "=" * 70)
    print("FRONTEND INTEGRATION GUIDE")
    print("=" * 70)
    
    print("""
📋 EXACT REQUEST FORMAT:

const response = await fetch('http://localhost:8000/api/practice/interview/start-round', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    round_type: "TECHNICAL_ROUND_1",  // REQUIRED (uppercase with underscores)
    domain: "Python",                  // REQUIRED (string)
    experience_years: 3,               // REQUIRED (integer 0-30)
    company_specific: "Google"         // OPTIONAL (string)
  })
});

⚠️ COMMON MISTAKES:
1. ❌ round_type: "Technical Round 1"  → ✅ "TECHNICAL_ROUND_1"
2. ❌ Missing domain field
3. ❌ experience_years as string → Must be integer
4. ❌ Lowercase enum values → Must be UPPERCASE

📊 GET DIFFICULTY BADGE:
Before showing rounds, call:
GET /api/practice/difficulty-preview?experience_years=3

Response: { "difficulty": "medium", "label": "MEDIUM" }
Then show "MEDIUM" badge on all round cards.
""")


if __name__ == "__main__":
    print("🚀 Starting Round-Based Interview API Tests\n")
    
    # Test valid requests
    test_valid_requests()
    
    # Test invalid requests
    test_invalid_requests()
    
    # Test difficulty preview
    test_difficulty_preview()
    
    # Print integration guide
    print_frontend_integration_guide()
    
    print("\n✅ All tests completed!\n")
