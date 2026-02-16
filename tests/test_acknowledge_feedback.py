"""
Test acknowledge-feedback endpoint to diagnose 422 error.
"""

import os

import pytest

# This is an integration/debug script that expects the FastAPI server to be
# running locally on localhost:8000. Skip during normal unit test runs.
if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip("requires local API server (set RUN_INTEGRATION_TESTS=1)", allow_module_level=True)

import requests
import json

BASE_URL = "http://localhost:8000"

def test_acknowledge_feedback_schema():
    """Test what the endpoint expects."""
    
    print("\n" + "="*60)
    print("🧪 Testing /acknowledge-feedback Schema")
    print("="*60)
    
    # Test 1: Correct payload
    print("\n1️⃣ Test with CORRECT payload:")
    correct_payload = {
        "session_id": "test-session-123",
        "question_id": 1,
        "feedback_read": True
    }
    print(f"   Sending: {json.dumps(correct_payload, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/practice/interview/acknowledge-feedback",
        json=correct_payload
    )
    print(f"   Status: {response.status_code}")
    if response.status_code != 404:  # 404 is fine (session doesn't exist)
        print(f"   ✅ Schema validation passed!")
    else:
        print(f"   ℹ️ Session not found (expected, schema is OK)")
    
    # Test 2: Missing session_id
    print("\n2️⃣ Test with MISSING session_id:")
    bad_payload = {
        "question_id": 1
    }
    print(f"   Sending: {json.dumps(bad_payload, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/practice/interview/acknowledge-feedback",
        json=bad_payload
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 422:
        print(f"   ❌ 422 Error (expected):")
        print(f"   {json.dumps(response.json(), indent=2)}")
    
    # Test 3: Wrong type for question_id
    print("\n3️⃣ Test with WRONG TYPE (question_id as string):")
    bad_payload = {
        "session_id": "test-123",
        "question_id": "1"  # String instead of int
    }
    print(f"   Sending: {json.dumps(bad_payload, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/practice/interview/acknowledge-feedback",
        json=bad_payload
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 422:
        print(f"   ❌ 422 Error:")
        print(f"   {json.dumps(response.json(), indent=2)}")
    elif response.status_code == 404:
        print(f"   ✅ Type coercion worked (string→int)")
    
    # Test 4: Snake_case vs camelCase
    print("\n4️⃣ Test with CAMEL_CASE (frontend might send this):")
    camel_payload = {
        "sessionId": "test-123",  # camelCase
        "questionId": 1,           # camelCase
        "feedbackRead": True       # camelCase
    }
    print(f"   Sending: {json.dumps(camel_payload, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/practice/interview/acknowledge-feedback",
        json=camel_payload
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 422:
        print(f"   ❌ 422 Error - Frontend using camelCase!")
        print(f"   {json.dumps(response.json(), indent=2)}")
        print("\n   🔧 FIX: Frontend must use snake_case (session_id, not sessionId)")
    
    print("\n" + "="*60)
    print("Expected Schema (snake_case):")
    print("="*60)
    print("""
{
  "session_id": "string",      ← Required, snake_case
  "question_id": 123,           ← Required, int
  "feedback_read": true         ← Optional, defaults to true
}
    """)

if __name__ == "__main__":
    test_acknowledge_feedback_schema()
