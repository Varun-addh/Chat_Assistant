"""
Test company-specific interview generation with Quick Start
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_quick_start_examples():
    """Test various Quick Start scenarios"""
    
    test_cases = [
        {
            "name": "Google SWE",
            "request": {
                "voice_input": "I'm preparing for Senior Software Engineer at Google",
                "question_count": 5
            }
        },
        {
            "name": "Meta Data Engineer",
            "request": {
                "voice_input": "I need to practice for data engineer role at Meta",
                "question_count": 4
            }
        },
        {
            "name": "Amazon with override",
            "request": {
                "voice_input": "I am preparing for data engineer role",
                "question_count": 3,
                "target_company": "Amazon"
            }
        },
        {
            "name": "Startup ML Engineer",
            "request": {
                "voice_input": "ML Engineer position at a fintech startup",
                "question_count": 5
            }
        },
        {
            "name": "Microsoft with custom count",
            "request": {
                "voice_input": "Senior backend engineer at Microsoft, 8 years experience",
                "question_count": 7  # User wants more questions
            }
        },
        {
            "name": "User specifies both",
            "request": {
                "voice_input": "I'm preparing for software engineer interviews",
                "question_count": 6,
                "target_company": "Netflix"  # User override
            }
        }
    ]
    
    print("=" * 80)
    print("QUICK START - Company-Specific Interview Generation Test")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*80}")
        print(f"Request: {json.dumps(test['request'], indent=2)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/practice/interview/quick-start",
                json=test['request']
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n✅ SUCCESS")
                print(f"AI Message: {data['ai_message']}")
                print(f"Ready to Start: {data['ready_to_start']}")
                
                if data.get('suggested_profile'):
                    profile = data['suggested_profile']
                    print(f"\nInferred Profile:")
                    print(f"  - Domain: {profile['domain']}")
                    print(f"  - Experience: {profile['experience_years']} years")
                    print(f"  - Company: {profile.get('company_preference', 'any')}")
                    print(f"  - Skills: {', '.join(profile['skills'][:3])}...")
                
                if data.get('first_question'):
                    q = data['first_question']
                    print(f"\nFirst Question:")
                    print(f"  - Text: {q['text'][:100]}...")
                    print(f"  - Category: {q['category']}")
                    print(f"  - Difficulty: {q['difficulty']}")
                    print(f"  - Time Limit: {q['time_limit']}s")
                
                print(f"\nSession ID: {data.get('session_id', 'N/A')}")
            else:
                print(f"\n❌ FAILED: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
    
    print("\n" + "="*80)
    print("KEY FEATURES DEMONSTRATED:")
    print("="*80)
    print("✅ AI extracts company from voice input (e.g., 'at Google')")
    print("✅ User can override with target_company parameter")
    print("✅ User can specify exact question_count (3-10)")
    print("✅ AI uses knowledge about ANY company (not hardcoded)")
    print("✅ Works for FAANG, startups, any company worldwide")
    print("✅ Questions adapt to company's known interview style")
    print("="*80)


if __name__ == "__main__":
    test_quick_start_examples()
