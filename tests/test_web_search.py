"""
Test real-time web search for interview questions
"""
import os

import pytest

# This file is a manual/integration diagnostic script. It requires network
# access and may depend on optional search providers. Skip during normal pytest
# runs unless explicitly enabled.
if __name__ != "__main__" and os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "web search integration test (set RUN_INTEGRATION_TESTS=1 to run)",
        allow_module_level=True,
    )

import asyncio
from app.services.chat.dynamic_interview_sources import WebSearchAdapter, QuestionDomain

async def test_web_search():
    """Test web search for real interview questions"""
    
    adapter = WebSearchAdapter()
    
    print("=" * 80)
    print("TESTING REAL-TIME WEB SEARCH FOR INTERVIEW QUESTIONS")
    print("=" * 80)
    
    # Test 1: DevOps questions
    print("\n1. Searching for: 'Docker vs Kubernetes' (DevOps)")
    questions = await adapter.search(
        query="Docker vs Kubernetes",
        domain=QuestionDomain.DEVOPS,
        company=None,
        limit=3
    )
    
    for i, q in enumerate(questions, 1):
        print(f"\n   Question {i}:")
        print(f"   Q: {q.question}")
        print(f"   Source: {q.source_platform}")
        print(f"   URL: {q.source_url}")
        print(f"   Credibility: {q.credibility_score}")
        print(f"   Answer Preview: {q.answer[:200]}...")
    
    # Test 2: AWS questions
    print("\n" + "=" * 80)
    print("\n2. Searching for: 'AWS Lambda' (Cloud)")
    questions = await adapter.search(
        query="AWS Lambda",
        domain=QuestionDomain.CLOUD,
        company="Amazon",
        limit=3
    )
    
    for i, q in enumerate(questions, 1):
        print(f"\n   Question {i}:")
        print(f"   Q: {q.question}")
        print(f"   Source: {q.source_platform}")
        print(f"   URL: {q.source_url}")
        print(f"   Credibility: {q.credibility_score}")
        print(f"   Answer Preview: {q.answer[:200]}...")
    
    # Test 3: System Design
    print("\n" + "=" * 80)
    print("\n3. Searching for: 'Design Twitter' (System Design)")
    questions = await adapter.search(
        query="Design Twitter",
        domain=QuestionDomain.SYSTEM_DESIGN,
        company=None,
        limit=3
    )
    
    for i, q in enumerate(questions, 1):
        print(f"\n   Question {i}:")
        print(f"   Q: {q.question}")
        print(f"   Source: {q.source_platform}")
        print(f"   URL: {q.source_url}")
        print(f"   Credibility: {q.credibility_score}")
        print(f"   Answer Preview: {q.answer[:200]}...")
    
    await adapter.close()
    
    print("\n" + "=" * 80)
    print("WEB SEARCH TEST COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_web_search())
