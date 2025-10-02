#!/usr/bin/env python3
"""
Simple test for code formatting
"""

# Test the problematic text
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

print("Testing code detection...")

# Test if this would be detected as code
def is_code_content(text):
    import re
    
    # Check for code block markers
    if '```' in text:
        return True
    
    # Check for Python-specific patterns
    python_patterns = [
        r'def\s+\w+\s*\(',  # Function definitions
        r'class\s+\w+',  # Class definitions
        r'import\s+\w+',  # Import statements
        r'from\s+\w+\s+import',  # From imports
        r'if\s+__name__\s*==\s*["\']__main__["\']',  # Main guard
        r'return\s+',  # Return statements
        r'while\s+',  # While loops
        r'for\s+\w+\s+in\s+',  # For loops
        r'#\s*[A-Z]',  # Comments starting with capital letters
    ]
    
    # Check if any Python patterns are found
    for pattern in python_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    # Check for indented code blocks (4+ spaces at start of line)
    lines = text.split('\n')
    indented_lines = 0
    for line in lines:
        if line.strip() and line.startswith('    '):
            indented_lines += 1
    
    # If more than 30% of non-empty lines are indented, it's likely code
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines and indented_lines / len(non_empty_lines) > 0.3:
        return True
    
    return False

def clean_code_formatting(text):
    import re
    
    # Fix common indentation issues
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Fix lines that look like they were formatted as table rows
        # Pattern: |variable = value|# comment|
        if '|' in line and '=' in line:
            # Remove table formatting and fix indentation
            line = re.sub(r'^\s*\|\s*', '', line)  # Remove leading | and spaces
            line = re.sub(r'\s*\|\s*$', '', line)  # Remove trailing | and spaces
            line = re.sub(r'\s*\|\s*', ' ', line)  # Replace middle | with spaces
            
            # Fix indentation for code lines
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # This looks like a code line that needs indentation
                if any(keyword in line for keyword in ['def ', 'class ', 'if ', 'while ', 'for ', 'else:', 'elif ']):
                    # This is a top-level statement, no indentation needed
                    pass
                elif line.strip().startswith(('return', 'yield', 'break', 'continue', 'pass')):
                    # This should be indented
                    line = '    ' + line.strip()
                elif '=' in line and not line.strip().startswith('#'):
                    # This looks like a variable assignment that should be indented
                    line = '    ' + line.strip()
        
        # Fix comment formatting
        if '|' in line and '#' in line:
            # Convert table-formatted comments to proper comments
            line = re.sub(r'^\s*\|\s*', '', line)
            line = re.sub(r'\s*\|\s*$', '', line)
            line = re.sub(r'\s*\|\s*', ' ', line)
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# Test the functions
print("Is code content?", is_code_content(problematic_text))

if is_code_content(problematic_text):
    print("✅ Code detected correctly!")
    formatted = clean_code_formatting(problematic_text)
    print("\nFormatted result:")
    print("=" * 50)
    print(formatted)
    print("=" * 50)
else:
    print("❌ Code not detected - would be formatted as table")
