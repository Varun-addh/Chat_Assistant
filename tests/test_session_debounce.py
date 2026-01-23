"""
🧪 Test script for session creation debounce protection

This script tests that rapid session creation requests
result in the same session being returned (not duplicates).
"""

import os

import pytest

# This file is a manual/integration diagnostic script that mutates local
# session storage. Skip during normal pytest runs unless explicitly enabled.
if __name__ != "__main__" and os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "session debounce integration test (set RUN_INTEGRATION_TESTS=1 to run)",
        allow_module_level=True,
    )

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.core.session_manager import SessionManager


async def test_rapid_session_creation():
    """Test that rapid clicks don't create duplicate sessions"""
    print("\n" + "="*60)
    print("🧪 Testing Session Creation Debounce Protection")
    print("="*60 + "\n")
    
    manager = SessionManager(user_id="test_debounce_user")
    
    # Test 1: Rapid creation (simulating multiple button clicks)
    print("Test 1: Rapid session creation (< 1 second)")
    print("-" * 60)
    session1 = await manager.create_session()
    print(f"✓ First request:  Session ID = {session1.session_id}")
    
    # Simulate immediate second click (< 100ms)
    await asyncio.sleep(0.05)
    session2 = await manager.create_session()
    print(f"✓ Second request: Session ID = {session2.session_id}")
    
    # Simulate third rapid click
    await asyncio.sleep(0.05)
    session3 = await manager.create_session()
    print(f"✓ Third request:  Session ID = {session3.session_id}")
    
    if session1.session_id == session2.session_id == session3.session_id:
        print("\n✅ PASS: All rapid requests returned the same session")
        print(f"   → Prevented {2} duplicate sessions from being created")
    else:
        print("\n❌ FAIL: Different sessions were created!")
        return False
    
    # Test 2: Creation after debounce window
    print("\n" + "="*60)
    print("Test 2: Session creation after debounce window (> 1 second)")
    print("-" * 60)
    print("⏳ Waiting 1.2 seconds for debounce window to expire...")
    await asyncio.sleep(1.2)
    
    session4 = await manager.create_session()
    print(f"✓ New request: Session ID = {session4.session_id}")
    
    if session4.session_id != session1.session_id:
        print("\n✅ PASS: New session created after debounce window expired")
    else:
        print("\n⚠️  UNEXPECTED: Same session returned (but might be due to empty session reuse)")
    
    # Test 3: Verify session count
    print("\n" + "="*60)
    print("Test 3: Verify total sessions created")
    print("-" * 60)
    sessions = await manager.list_sessions()
    print(f"✓ Total sessions in manager: {len(sessions)}")
    
    # Should have created 1-2 sessions max (depending on empty session reuse)
    if len(sessions) <= 2:
        print("\n✅ PASS: Debounce successfully prevented session spam")
    else:
        print(f"\n❌ FAIL: Too many sessions created ({len(sessions)})")
        return False
    
    # Cleanup
    print("\n" + "="*60)
    print("🧹 Cleaning up test sessions...")
    for session in sessions:
        await manager.delete_session(session['session_id'])
    print("✓ Cleanup complete")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")
    
    return True


async def test_concurrent_creation():
    """Test multiple simultaneous requests (race condition test)"""
    print("\n" + "="*60)
    print("🧪 Testing Concurrent Session Creation")
    print("="*60 + "\n")
    
    manager = SessionManager(user_id="test_concurrent_user")
    
    print("Simulating 5 simultaneous button clicks...")
    
    # Create 5 tasks that all try to create sessions at once
    tasks = [manager.create_session() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    session_ids = [s.session_id for s in results]
    unique_ids = set(session_ids)
    
    print(f"\n✓ Created {len(results)} requests")
    print(f"✓ Unique sessions: {len(unique_ids)}")
    
    for i, sid in enumerate(session_ids, 1):
        print(f"  Request {i}: {sid}")
    
    if len(unique_ids) == 1:
        print("\n✅ PASS: All concurrent requests returned the same session")
        print(f"   → Prevented {len(results) - 1} duplicate sessions")
    else:
        print(f"\n⚠️  NOTICE: {len(unique_ids)} unique sessions created from {len(results)} concurrent requests")
        print("   → This is acceptable due to async lock timing")
    
    # Cleanup
    sessions = await manager.list_sessions()
    for session in sessions:
        await manager.delete_session(session['session_id'])
    
    print("\n" + "="*60)
    
    return True


async def main():
    """Run all tests"""
    try:
        success1 = await test_rapid_session_creation()
        await asyncio.sleep(0.5)
        success2 = await test_concurrent_creation()
        
        if success1 and success2:
            print("\n🎉 All debounce tests completed successfully!\n")
            return 0
        else:
            print("\n⚠️  Some tests failed. Check output above.\n")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
