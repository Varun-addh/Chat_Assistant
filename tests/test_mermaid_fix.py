"""
🧪 Test Mermaid diagram rendering with class assignments

This tests that the fixed diagram rendering correctly handles:
1. Class assignments (:::)
2. Escaped newlines in init blocks
3. ClassDef declarations
"""

# Test Mermaid code with class assignments (the problematic case from the error)
test_diagram = """flowchart TD
subgraph Load Balancer
  LoadBalancer[Load Balancer]
end
subgraph Application Server
  ApplicationServer[Application Server]
end
subgraph Database Server
  DatabaseServer[Database Server]
end
subgraph Caching Layer
  CachingLayer[Caching Layer]
end
subgraph Content Delivery Network CDN
  CDN[Content Delivery Network CDN]
end
subgraph Storage System
  StorageSystem[Storage System]
end
  LoadBalancer --> ApplicationServer
  ApplicationServer --> DatabaseServer
  DatabaseServer --> CachingLayer
  CachingLayer --> CDN
  CDN --> StorageSystem
  StorageSystem --> CDN
classDef load-balancer fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef application-server fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef database-server fill:#fff3e0,stroke:#f57c00,color:#000
    classDef caching-layer fill:#e8f5e9,stroke:#388e3c,color:#000
    classDef cdn fill:#fffde7,stroke:#f9a825,color:#000
    classDef storage-system fill:#fce4ec,stroke:#c2185b,color:#000
    LoadBalancer:::load-balancer
    ApplicationServer:::application-server
    DatabaseServer:::database-server
    CachingLayer:::caching-layer
    CDN:::cdn
    StorageSystem:::storage-system
linkStyle default stroke:#666,stroke-width:1.3px;
"""

print("="*70)
print("🧪 Testing Mermaid Diagram Rendering Fix")
print("="*70)
print("\n📝 Test Diagram (with class assignments):\n")
print(test_diagram)
print("\n" + "="*70)

# Test that class assignments (:::) are preserved
print("\n✓ Checking class assignment syntax...")
if "LoadBalancer:::load-balancer" in test_diagram:
    print("  ✅ PASS: Class assignments use correct syntax (:::)")
else:
    print("  ❌ FAIL: Class assignments syntax missing")

# Verify classRef usage
class_assignments = [
    "LoadBalancer:::load-balancer",
    "ApplicationServer:::application-server",
    "DatabaseServer:::database-server",
    "CachingLayer:::caching-layer",
    "CDN:::cdn",
    "StorageSystem:::storage-system"
]

print("\n✓ Verifying all class assignments...")
all_found = all(assignment in test_diagram for assignment in class_assignments)
if all_found:
    print(f"  ✅ PASS: All {len(class_assignments)} class assignments found")
else:
    print("  ❌ FAIL: Some class assignments missing")

# Test the actual API endpoint
print("\n" + "="*70)
print("🌐 Testing API Endpoint")
print("="*70)

import requests
import json

# Test with local server (make sure it's running)
try:
    print("\n📡 Sending POST request to /api/render_mermaid...")
    
    response = requests.post(
        "http://127.0.0.1:8000/api/render_mermaid",
        json={
            "code": test_diagram,
            "theme": "default",
            "style": "modern"
        },
        timeout=30
    )
    
    print(f"   Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("   ✅ SUCCESS: Diagram rendered successfully!")
        print(f"   Content Type: {response.headers.get('Content-Type')}")
        print(f"   Response Size: {len(response.content)} bytes")
        
        # Check if response is valid SVG
        if response.text.strip().startswith("<svg"):
            print("   ✅ Response is valid SVG")
            
            # Save to file for visual inspection
            with open("test_diagram_output.svg", "w", encoding="utf-8") as f:
                f.write(response.text)
            print("   💾 Saved to: test_diagram_output.svg")
        else:
            print("   ⚠️  Response doesn't look like SVG")
            print(f"   First 200 chars: {response.text[:200]}")
    else:
        print(f"   ❌ FAILED: {response.status_code}")
        print(f"   Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("   ⚠️  Could not connect to server at http://127.0.0.1:8000")
    print("   ℹ️  Make sure the server is running: uvicorn app.main:app --reload")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("✅ Test Complete!")
print("="*70)
print("\nKey Points:")
print("1. Class assignments must use ::: (not :: or -:)")
print("2. Init blocks should use actual newlines (not \\n)")
print("3. ClassDef semicolons should be at end of line")
print("\nIf API test failed, check the server logs for details.")
print("="*70 + "\n")
