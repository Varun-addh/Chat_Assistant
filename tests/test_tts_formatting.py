"""
Test TTS text formatting to see how questions are enhanced for natural delivery.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.practice.interviewer_agent import InterviewerAgent
from app.schemas import PracticeInterviewQuestion, QuestionDifficulty, SpeechAnalyticsConfig


def test_formatting():
    """Test how questions are formatted for TTS."""
    
    # Initialize agent (with dummy API key since we're only testing formatting)
    agent = InterviewerAgent(
        analytics_config=SpeechAnalyticsConfig(),
        gemini_api_key="dummy-key-for-testing"
    )
    
    # Test questions
    test_questions = [
        PracticeInterviewQuestion(
            id=1,
            text="Tell me about your experience with Python",
            difficulty=QuestionDifficulty.EASY,
            category="technical",
            time_limit=90
        ),
        PracticeInterviewQuestion(
            id=2,
            text="How do you handle conflict in a team",
            difficulty=QuestionDifficulty.MEDIUM,
            category="behavioral",
            time_limit=90
        ),
        PracticeInterviewQuestion(
            id=3,
            text="What are the differences between REST and GraphQL APIs and when would you use each",
            difficulty=QuestionDifficulty.HARD,
            category="technical",
            time_limit=120
        ),
        PracticeInterviewQuestion(
            id=4,
            text="Describe a time when you had to learn a new technology quickly",
            difficulty=QuestionDifficulty.MEDIUM,
            category="behavioral",
            time_limit=90
        ),
        PracticeInterviewQuestion(
            id=5,
            text="Where do you see yourself in 5 years",
            difficulty=QuestionDifficulty.EASY,
            category="behavioral",
            time_limit=60
        ),
    ]
    
    print("=" * 80)
    print("TTS TEXT FORMATTING TEST")
    print("=" * 80)
    print()
    print("This shows how questions are formatted for natural TTS delivery.")
    print("Note the added punctuation for pauses and intonation!")
    print()
    print("=" * 80)
    print()
    
    for q in test_questions:
        print(f"QUESTION {q.id} ({q.category.upper()}, {q.difficulty.value})")
        print("-" * 80)
        print()
        
        print("📝 ORIGINAL TEXT:")
        print(f"   {q.text}")
        print()
        
        formatted = agent.format_tts_text(q)
        print("🎙️ TTS FORMATTED TEXT:")
        print(f"   {formatted}")
        print()
        
        print("💡 WHAT TTS WILL DO:")
        print("   - Intro creates conversational opening")
        print("   - Commas (,) = brief pause for breathing")
        print("   - Ellipsis (...) = thinking pause")
        print("   - Question mark (?) = rising intonation at end")
        if q.category.lower() in ['behavioral', 'situational']:
            print("   - Behavioral prompt helps candidate prepare")
        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    test_formatting()
