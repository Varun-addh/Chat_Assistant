import sys
import os

class MockSettings:
    app_name = "Stratax AI"
    app_developer_name = "Varun"

s = MockSettings()

from app.services.llm.identity import is_identity_question

tests = [
    ("what is mirror mode in stratax?", False),
    ("what is stratax AI meant to be?", True),
    ("tell me about stratax", True),
    ("who developed you", True),
    ("what is stratax", True),
    ("what stratax is", True),
    ("who developed the mirror mode in stratax?", True)
]

passed = True
for q, expected in tests:
    res = is_identity_question(s, q)
    if res != expected:
        print(f"FAILED: '{q}' -> expected {expected}, got {res}")
        passed = False
    else:
        print(f"PASSED: '{q}' -> {res}")

if passed:
    print("ALL TESTS PASSED")
