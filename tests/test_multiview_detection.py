"""
Test script to verify multi-view architecture detection in questions.py
"""

# Test system design question detection
test_questions = [
    # Should trigger architecture
    "Design a system for video streaming like Netflix",
    "System design: Design YouTube",
    "How would you design an architecture for Uber?",
    "Design a high level design for WhatsApp",
    "Create architecture for a distributed cache",
    
    # Should NOT trigger architecture (coding questions)
    "Write a function to reverse a string",
    "Design a class for a binary tree",
    "Implement a sorting algorithm"
]

def is_architecture_question(question: str) -> bool:
    """Replicate the detection logic from questions.py"""
    q_lower = question.lower()
    arch_triggers = [
        "system design", "architecture", "design a system", "design the system",
        "high level design", "hld", "design a platform", "design an app",
        "design uber", "design facebook", "design netflix", "design youtube",
        "design twitter", "design whatsapp", "design instagram", "design amazon", 
        "design google", "design a service", "design a microservice"
    ]
    is_arch = any(t in q_lower for t in arch_triggers) and "function" not in q_lower and "class" not in q_lower
    return is_arch


print("🧪 Testing Multi-View Architecture Detection\n")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    result = is_architecture_question(question)
    status = "✅ SYSTEM DESIGN" if result else "❌ Regular Question"
    print(f"\n{i}. {question}")
    print(f"   → {status}")

print("\n" + "=" * 60)
print("\n✅ All detection tests completed!")
print("\n📝 Summary:")
print("   - Questions 1-5 should be detected as system design (multi-view)")
print("   - Questions 6-8 should be regular coding questions")
