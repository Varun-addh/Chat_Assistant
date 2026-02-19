"""
Diagnostic script to test evaluation generation.
Run this after completing an interview to check if evaluation is working.
"""

import os

import pytest

# This file is a manual/integration diagnostic script. It depends on external
# LLM credentials and network access. Skip during normal pytest runs unless
# explicitly enabled.
if __name__ != "__main__" and os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "evaluation integration test (set RUN_INTEGRATION_TESTS=1 to run)",
        allow_module_level=True,
    )

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.practice.evaluation_agent import EvaluationAgent
from app.schemas import (
    AnswerSubmission,
    SpeechMetrics,
    MicroFeedback
)
from app.config import get_settings


async def test_evaluation():
    """Test evaluation with sample data."""
    settings = get_settings()
    
    # Initialize evaluation agent
    agent = EvaluationAgent(
        api_key=settings.GEMINI_API_KEY,
        model_name="gemini-1.5-pro"
    )
    
    # Create sample answers
    sample_answers = [
        AnswerSubmission(
            question_id=1,
            transcript="I have five years of experience in Python backend development. I've worked with FastAPI, Django, and Flask frameworks. My recent project involved building a microservices architecture using Docker and Kubernetes.",
            metrics=SpeechMetrics(
                filler_count=2,
                wpm=145.0,
                longest_silence=1.5,
                confidence_score=7.5,
                overtalked=False,
                duration=25.0,
                filler_words=["um", "uh"],
                pause_count=1,
                pitch_variance=4500.0
            ),
            micro_feedback=MicroFeedback(
                delivery_tips=["Good pace", "Clear speech"],
                pace_feedback="Excellent speaking pace at 145 WPM",
                overall_note="Strong technical answer with good examples"
            ),
            audio_duration=25.0
        ),
        AnswerSubmission(
            question_id=2,
            transcript="My greatest strength is problem solving. I like to break down complex problems into smaller parts. For example, when we had a performance issue, I used profiling tools to identify the bottleneck.",
            metrics=SpeechMetrics(
                filler_count=1,
                wpm=150.0,
                longest_silence=2.0,
                confidence_score=8.0,
                overtalked=False,
                duration=22.0,
                filler_words=["like"],
                pause_count=2,
                pitch_variance=5200.0
            ),
            micro_feedback=MicroFeedback(
                delivery_tips=["Confident delivery"],
                pace_feedback="Good pace at 150 WPM",
                overall_note="Clear answer with specific example"
            ),
            audio_duration=22.0
        ),
        AnswerSubmission(
            question_id=3,
            transcript="Um, so like, I handle challenging situations by, you know, staying calm and, uh, analyzing the problem. Like when we had a production outage, I basically coordinated with the team and, um, we fixed it quickly.",
            metrics=SpeechMetrics(
                filler_count=8,
                wpm=135.0,
                longest_silence=3.5,
                confidence_score=5.5,
                overtalked=False,
                duration=28.0,
                filler_words=["um", "so", "like", "you know", "uh", "like", "basically", "um"],
                pause_count=3,
                pitch_variance=3200.0
            ),
            micro_feedback=MicroFeedback(
                delivery_tips=["Reduce filler words", "More confident tone"],
                pace_feedback="Slightly slow at 135 WPM",
                overall_note="Answer needs more clarity and fewer fillers"
            ),
            audio_duration=28.0
        ),
        AnswerSubmission(
            question_id=4,
            transcript="I'm most proud of the API gateway project I led. We reduced latency by 60% and improved reliability. I designed the architecture, coordinated with three teams, and delivered it two weeks ahead of schedule.",
            metrics=SpeechMetrics(
                filler_count=0,
                wpm=155.0,
                longest_silence=1.2,
                confidence_score=9.0,
                overtalked=False,
                duration=20.0,
                filler_words=[],
                pause_count=1,
                pitch_variance=6100.0
            ),
            micro_feedback=MicroFeedback(
                delivery_tips=["Excellent answer"],
                pace_feedback="Perfect pace at 155 WPM",
                overall_note="Outstanding answer with quantifiable results"
            ),
            audio_duration=20.0
        ),
        AnswerSubmission(
            question_id=5,
            transcript="In five years I see myself as a technical lead or architect. I want to continue growing my skills in distributed systems and cloud technologies. I'm particularly interested in contributing to open source projects and mentoring junior developers.",
            metrics=SpeechMetrics(
                filler_count=0,
                wpm=148.0,
                longest_silence=1.8,
                confidence_score=7.8,
                overtalked=False,
                duration=24.0,
                filler_words=[],
                pause_count=2,
                pitch_variance=5000.0
            ),
            micro_feedback=MicroFeedback(
                delivery_tips=["Clear vision"],
                pace_feedback="Good pace at 148 WPM",
                overall_note="Well-structured career goal answer"
            ),
            audio_duration=24.0
        )
    ]
    
    print("🧪 Testing Evaluation Agent...")
    print(f"Sample answers: {len(sample_answers)}")
    print()
    
    try:
        # Generate evaluation
        print("📊 Generating evaluation...")
        evaluation = await agent.evaluate_interview(sample_answers, "test-session-001")
        
        print("\n✅ Evaluation Generated Successfully!")
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print("\n💪 STRENGTHS:")
        for i, strength in enumerate(evaluation.strengths.items, 1):
            print(f"  {i}. {strength}")
        
        print("\n📈 IMPROVEMENTS:")
        for i, improvement in enumerate(evaluation.improvements.items, 1):
            print(f"  {i}. {improvement}")
        
        print("\n📊 METRICS SUMMARY:")
        print(f"  Total Fillers: {evaluation.metrics_summary.total_fillers}")
        print(f"  Average WPM: {evaluation.metrics_summary.avg_wpm}")
        print(f"  Longest Pause: {evaluation.metrics_summary.longest_pause}s")
        print(f"  Average Confidence: {evaluation.metrics_summary.avg_confidence}/10")
        print(f"  Total Duration: {evaluation.metrics_summary.total_duration}s")
        print(f"  Overtalked Count: {evaluation.metrics_summary.overtalked_count}")
        
        print("\n🎯 ACTION PLAN:")
        for i, step in enumerate(evaluation.action_plan.steps, 1):
            print(f"  {i}. {step}")
        
        print(f"\n🎓 RECOMMENDATION:")
        print(f"  {evaluation.practice_recommendation}")
        
        print("\n" + "="*60)
        
        # Save to file for inspection
        output_file = Path("test_evaluation_output.json")
        with open(output_file, 'w') as f:
            json.dump(evaluation.dict(), f, indent=2, default=str)
        print(f"\n💾 Full evaluation saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Evaluation Failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("EVALUATION AGENT DIAGNOSTIC TEST")
    print("="*60)
    print()
    
    success = asyncio.run(test_evaluation())
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed - check error messages above")
        sys.exit(1)
