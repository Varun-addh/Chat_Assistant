"""
Debug web search to see what results we're getting
"""

import os

import pytest

# This file is a manual/integration debug script. It requires optional network
# access and the duckduckgo_search package, which may not be installed in CI.
if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip("web search debug integration test (set RUN_INTEGRATION_TESTS=1 to run)", allow_module_level=True)

import asyncio

duckduckgo_search = pytest.importorskip("duckduckgo_search")
DDGS = duckduckgo_search.DDGS

async def test_ddg():
    """Test DuckDuckGo search directly"""
    
    print("Testing DuckDuckGo search directly...")
    
    query = "Docker vs Kubernetes interview questions 2024"
    print(f"\nQuery: {query}")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('title', 'No title')}")
            print(f"   URL: {result.get('href', 'No URL')}")
            print(f"   Body: {result.get('body', 'No body')[:150]}...")
            print()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ddg())
