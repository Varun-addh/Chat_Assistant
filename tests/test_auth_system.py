"""
Test the new authentication and rate limiting system
"""
from __future__ import annotations

# NOTE: This file is an integration smoke-test that requires a running server.
# Pytest will try to collect it (because it starts with test_), but it is not
# meant to run in CI/unit-test mode.
if __name__ != "__main__":
    import pytest
    pytest.skip(
        "Skipping integration-only auth smoke test (requires a running server). Run this file directly to execute.",
        allow_module_level=True,
    )

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


async def test_auth_flow():
    """Test registration, login, and protected routes"""
    print("="*60)
    print("🧪 TESTING AUTHENTICATION & RATE LIMITING SYSTEM")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        # 1. Test health endpoint (no auth required)
        print("\n1️⃣  Testing health endpoint (public)...")
        response = await client.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # 2. Register a new user
        print("\n2️⃣  Registering new user...")
        timestamp = int(datetime.now().timestamp())
        test_email = f"test_{timestamp}@stratax.ai"
        register_data = {
            "email": test_email,
            "password": "SecurePass123!",  # Shorter password (under 72 bytes)
            "full_name": "Test User",
            "username": f"user_{timestamp}"
        }
        
        response = await client.post(f"{BASE_URL}/auth/register", json=register_data)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 201:
            auth_data = response.json()
            token = auth_data["access_token"]
            print(f"   ✅ User registered successfully!")
            print(f"   User ID: {auth_data['user_id']}")
            print(f"   Tier: {auth_data['tier']}")
            print(f"   Token: {token[:50]}...")
        else:
            print(f"   ❌ Registration failed: {response.json()}")
            return
        
        # 3. Get user info
        print("\n3️⃣  Getting user info...")
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(f"{BASE_URL}/auth/me", headers=headers)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            user_info = response.json()
            print(f"   ✅ User info retrieved:")
            print(f"      Email: {user_info['email']}")
            print(f"      Tier: {user_info['tier']}")
            print(f"      Username: {user_info['username']}")
        
        # 4. Get quota info
        print("\n4️⃣  Checking quota limits...")
        response = await client.get(f"{BASE_URL}/auth/quota", headers=headers)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            quota = response.json()
            print(f"   ✅ Quota info:")
            print(f"      Tier: {quota['tier']}")
            print(f"      Daily API calls: {quota['limits']['daily_api_calls']}")
            print(f"      Daily copilot questions: {quota['limits']['daily_copilot_questions']}")
            print(f"      Daily mock interviews: {quota['limits']['daily_mock_interviews']}")
        
        # 5. Test protected route with auth
        print("\n5️⃣  Testing protected route (with auth)...")
        response = await client.post(
            f"{BASE_URL}/api/sessions/create",
            headers=headers,
            json={"user_id": auth_data['user_id']}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # 6. Test rate limiting by making multiple requests
        print("\n6️⃣  Testing rate limiting (making 5 rapid requests)...")
        for i in range(5):
            response = await client.get(f"{BASE_URL}/auth/me", headers=headers)
            remaining = response.headers.get("X-RateLimit-Remaining", "N/A")
            print(f"   Request {i+1}: Status={response.status_code}, Remaining={remaining}")
        
        # 7. Test login with existing user
        print("\n7️⃣  Testing login...")
        login_data = {
            "email": test_email,
            "password": "SecurePass123!"
        }
        response = await client.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Login successful!")
            new_token = response.json()["access_token"]
            print(f"   New Token: {new_token[:50]}...")
        
        # 8. Test invalid token
        print("\n8️⃣  Testing invalid token...")
        bad_headers = {"Authorization": "Bearer invalid_token_12345"}
        response = await client.get(f"{BASE_URL}/auth/me", headers=bad_headers)
        print(f"   Status: {response.status_code}")
        print(f"   Expected 401 Unauthorized: {'✅' if response.status_code == 401 else '❌'}")
        
        # 9. Test guest access (no token)
        print("\n9️⃣  Testing guest access (no token)...")
        response = await client.post(
            f"{BASE_URL}/api/sessions/create",
            json={}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    
    print("\n" + "="*60)
    print("✅ AUTHENTICATION TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    print(f"\n🚀 Make sure the server is running on {BASE_URL}")
    print("   Example: uvicorn app.main:app --reload --port 8000\n")
    
    input("Press Enter to start tests...")
    
    asyncio.run(test_auth_flow())
