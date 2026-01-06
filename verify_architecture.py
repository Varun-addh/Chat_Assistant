"""
Verify Architecture Generator
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/diagrams"

def run_test():
    print("🚀 Starting Architecture Generator Test...")
    
    # 1. Check available views
    print("\n1️⃣  Checking Available Views...")
    try:
        resp = requests.get(f"{BASE_URL}/architecture/available_views")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ Success! Found {data['total_views']} view types")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 2. Generate Architecture
    print("\n2️⃣  Generating Architecture (this takes ~30-60s)...")
    payload = {
        "system_description": "Ride sharing app like Uber with real-time tracking",
        "user_level": "mid",
        "style": "modern",
        "include_explanations": True
    }
    
    start_time = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/generate_architecture",
            json=payload,
            headers={"X-API-Key": "test-key-12345"} # Assuming this key from previous context
        )
        
        if resp.status_code == 200:
            data = resp.json()
            duration = time.time() - start_time
            print(f"   ✅ Success! Generated {data['total_views']} views in {duration:.1f}s")
            print(f"   📦 System Name: {data['system_name']}")
            print(f"   📊 Views: {[v['title'] for v in data['views']]}")
        else:
            print(f"   ❌ Failed: {resp.status_code}")
            print(f"   Response: {resp.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    run_test()
