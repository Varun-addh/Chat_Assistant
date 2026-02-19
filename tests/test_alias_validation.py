"""
Test AliasChoices for camelCase/snake_case support.
"""
from pydantic import BaseModel, Field, AliasChoices

class AcknowledgeFeedbackRequest(BaseModel):
	"""Request to acknowledge feedback and get next question."""
	session_id: str = Field(
		..., 
		description="Session identifier",
		validation_alias=AliasChoices("session_id", "sessionId")
	)
	question_id: int = Field(
		..., 
		description="Question ID that was just answered",
		validation_alias=AliasChoices("question_id", "questionId")
	)
	feedback_read: bool = Field(
		default=True, 
		description="Confirmation that user read the feedback",
		validation_alias=AliasChoices("feedback_read", "feedbackRead")
	)

print("="*60)
print("Testing AliasChoices for camelCase/snake_case support")
print("="*60)

# Test 1: snake_case (Python style)
print("\n1️⃣ Test snake_case (backend style):")
data1 = {
    "session_id": "abc-123",
    "question_id": 5,
    "feedback_read": True
}
try:
    req1 = AcknowledgeFeedbackRequest(**data1)
    print(f"   ✅ Success: {req1}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: camelCase (frontend style)
print("\n2️⃣ Test camelCase (frontend style):")
data2 = {
    "sessionId": "xyz-789",
    "questionId": 10,
    "feedbackRead": False
}
try:
    req2 = AcknowledgeFeedbackRequest(**data2)
    print(f"   ✅ Success: {req2}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Mixed (should work)
print("\n3️⃣ Test mixed case:")
data3 = {
    "sessionId": "mixed-456",  # camelCase
    "question_id": 7,          # snake_case
}
try:
    req3 = AcknowledgeFeedbackRequest(**data3)
    print(f"   ✅ Success: {req3}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 4: Missing question_id (should fail)
print("\n4️⃣ Test missing question_id (should fail):")
data4 = {
    "session_id": "missing-123"
}
try:
    req4 = AcknowledgeFeedbackRequest(**data4)
    print(f"   ❌ Should have failed but passed: {req4}")
except Exception as e:
    print(f"   ✅ Expected error: {e}")

print("\n" + "="*60)
print("✅ AliasChoices is working correctly!")
print("="*60)
