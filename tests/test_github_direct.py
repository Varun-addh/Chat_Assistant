import aiohttp
import asyncio
import os

import pytest

# This file is a manual/integration diagnostic script. It may require a GitHub
# token and network access. Skip during normal pytest runs unless explicitly
# enabled.
if __name__ != "__main__" and os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "github integration test (set RUN_INTEGRATION_TESTS=1 to run)",
        allow_module_level=True,
    )
from dotenv import load_dotenv

load_dotenv()

async def test_github():
    github_token = os.getenv("GITHUB_TOKEN")
    print(f"GitHub token loaded: {github_token[:20]}..." if github_token else "NO TOKEN")
    
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    headers["Accept"] = "application/vnd.github.v3+json"
    
    async with aiohttp.ClientSession(headers=headers) as session:
        # Test GitHub API
        search_url = "https://api.github.com/search/code"
        params = {
            "q": "machine learning interview questions extension:md stars:>100",
            "per_page": 5
        }
        
        print(f"\nTesting: {search_url}")
        print(f"Query: {params['q']}")
        print(f"Headers: {headers}")
        
        async with session.get(search_url, params=params, timeout=15) as response:
            print(f"\nStatus: {response.status}")
            print(f"Headers: {dict(response.headers)}")
            
            if response.status == 200:
                data = await response.json()
                print(f"\nTotal results: {data.get('total_count', 0)}")
                print(f"Items returned: {len(data.get('items', []))}")
                
                for i, item in enumerate(data.get("items", [])[:3]):
                    print(f"\n--- Result {i+1} ---")
                    print(f"Name: {item.get('name')}")
                    print(f"Path: {item.get('path')}")
                    print(f"Repo: {item.get('repository', {}).get('full_name')}")
                    print(f"URL: {item.get('html_url')}")
            else:
                text = await response.text()
                print(f"\nError response: {text[:500]}")

if __name__ == "__main__":
    asyncio.run(test_github())
