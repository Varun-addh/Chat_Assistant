#!/usr/bin/env python3
"""
Test script to verify the code formatting fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.llm_service import LLMService

def test_code_formatting():
    """Test the code formatting with the problematic linked list example"""
    
    # Create the LLM service instance
    llm_service = LLMService()
    
    # Test the problematic text that was being formatted as a table
    problematic_text = """class ListNode:
\"\"\"Node of a singly‑linked list.\"\"\"
    def __init__(self, val=0, next=None):
self.val = val
self.next = next


def reverse_linked_list(head: ListNode) -> ListNode:
    \"\"\"
    Reverses a singly‑linked list.

    Args:
        head: The first node of the list (or None for an empty list).

    Returns:
        The new head of the reversed list.
    \"\"\"
|prev = None|# Will become the new head|
|---|---|
|current = head|# Node we are currently processing|

    while current:
|nxt = current.next|# Store next node|
|---|---|
|current.next = prev|# Reverse the link|
|prev = current|# Move prev forward|
|current = nxt|# Move to next node|

    return prev"""

    print("Original problematic text:")
    print("=" * 50)
    print(problematic_text)
    print("=" * 50)
    
    # Test the formatting
    formatted_text = llm_service._format_response(problematic_text)
    
    print("\nFormatted text:")
    print("=" * 50)
    print(formatted_text)
    print("=" * 50)
    
    # Check if it's properly formatted
    if "|" not in formatted_text or "---" not in formatted_text:
        print("\n✅ SUCCESS: Code is no longer formatted as a table!")
    else:
        print("\n❌ FAILED: Code is still being formatted as a table")
    
    # Check for proper indentation
    lines = formatted_text.split('\n')
    has_proper_indentation = any(line.startswith('    ') for line in lines if line.strip())
    
    if has_proper_indentation:
        print("✅ SUCCESS: Code has proper indentation!")
    else:
        print("❌ WARNING: Code indentation might need improvement")

if __name__ == "__main__":
    test_code_formatting()
