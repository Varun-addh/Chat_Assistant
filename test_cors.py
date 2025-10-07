#!/usr/bin/env python3
"""
Simple CORS test script to verify the API accepts requests from the frontend domain.
"""
import requests
import json

# Test URLs
API_BASE = "https://chat-assistant-xeij.onrender.com"
FRONTEND_ORIGIN = "https://inverviewast.web.app"

def test_cors_preflight():
    """Test CORS preflight request"""
    print("Testing CORS preflight request...")
    
    headers = {
        "Origin": FRONTEND_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type,Authorization"
    }
    
    try:
        response = requests.options(f"{API_BASE}/api/question", headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        # Check for CORS headers
        cors_headers = {
            "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
            "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
            "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers"),
        }
        
        print(f"CORS Headers: {cors_headers}")
        
        if cors_headers["Access-Control-Allow-Origin"] == FRONTEND_ORIGIN or cors_headers["Access-Control-Allow-Origin"] == "*":
            print("✅ CORS preflight test PASSED")
            return True
        else:
            print("❌ CORS preflight test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ CORS preflight test FAILED with error: {e}")
        return False

def test_cors_actual_request():
    """Test actual CORS request"""
    print("\nTesting actual CORS request...")
    
    headers = {
        "Origin": FRONTEND_ORIGIN,
        "Content-Type": "application/json"
    }
    
    payload = {
        "question": "Test question",
        "session_id": "test-session"
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/question", 
                               headers=headers, 
                               json=payload)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        # Check for CORS headers in response
        cors_origin = response.headers.get("Access-Control-Allow-Origin")
        if cors_origin == FRONTEND_ORIGIN or cors_origin == "*":
            print("✅ CORS actual request test PASSED")
            return True
        else:
            print("❌ CORS actual request test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ CORS actual request test FAILED with error: {e}")
        return False

if __name__ == "__main__":
    print("CORS Test for Interview Assistant API")
    print("=" * 50)
    
    preflight_ok = test_cors_preflight()
    actual_ok = test_cors_actual_request()
    
    print("\n" + "=" * 50)
    if preflight_ok and actual_ok:
        print("🎉 All CORS tests PASSED!")
    else:
        print("💥 Some CORS tests FAILED!")
