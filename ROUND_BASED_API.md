# 🎯 Round-Based Interview Practice - API Documentation

## Overview

The Round-Based Interview Practice Mode allows users to practice specific interview rounds instead of generic questions. This mirrors real company interview processes and provides targeted preparation.

---

## Available Interview Rounds

### 1. **HR Screening Round** (`hr_screening`)
- **Duration:** 15-20 minutes
- **Questions:** 4
- **Difficulty:** Easy
- **Focus:** Background, motivation, culture fit
- **Categories:** Behavioral, motivation, background

### 2. **Technical Round 1 - Fundamentals** (`technical_round_1`)
- **Duration:** 30-45 minutes
- **Questions:** 6
- **Difficulty:** Medium
- **Focus:** Core concepts, DSA basics, fundamentals
- **Categories:** Technical, coding, fundamentals

### 3. **Technical Round 2 - Deep Dive** (`technical_round_2`)
- **Duration:** 45-60 minutes
- **Questions:** 6
- **Difficulty:** Hard
- **Focus:** Advanced concepts, architecture, problem-solving
- **Categories:** Technical, architecture, advanced coding

### 4. **System Design Round** (`system_design`)
- **Duration:** 45-60 minutes
- **Questions:** 2-3 (longer answers)
- **Difficulty:** Hard
- **Focus:** Scalability, distributed systems, tradeoffs
- **Categories:** System design, scalability, architecture

### 5. **Behavioral Round** (`behavioral`)
- **Duration:** 30-40 minutes
- **Questions:** 5
- **Difficulty:** Medium
- **Focus:** STAR method, teamwork, conflict resolution
- **Categories:** Behavioral, teamwork, leadership

### 6. **Managerial/Director Round** (`managerial`)
- **Duration:** 30-45 minutes
- **Questions:** 4
- **Difficulty:** Hard
- **Focus:** Strategic thinking, vision, leadership
- **Categories:** Leadership, strategy, vision

### 7. **Machine Learning Specialist** (`machine_learning`)
- **Duration:** 45-60 minutes
- **Questions:** 5
- **Difficulty:** Hard
- **Focus:** ML algorithms, models, deployment
- **Categories:** Machine learning, algorithms, ML systems

### 8. **Data Engineering Round** (`data_engineering`)
- **Duration:** 45-60 minutes
- **Questions:** 5
- **Difficulty:** Hard
- **Focus:** Data pipelines, ETL, big data
- **Categories:** Data engineering, pipelines, big data

### 9. **Frontend Specialist** (`frontend_specialist`)
- **Duration:** 45-60 minutes
- **Questions:** 6
- **Difficulty:** Hard
- **Focus:** React/Vue/Angular, performance, UI/UX
- **Categories:** Frontend, React, performance

### 10. **Backend Specialist** (`backend_specialist`)
- **Duration:** 45-60 minutes
- **Questions:** 6
- **Difficulty:** Hard
- **Focus:** APIs, databases, microservices
- **Categories:** Backend, APIs, databases

### 11. **DevOps/SRE Round** (`devops`)
- **Duration:** 45-60 minutes
- **Questions:** 5
- **Difficulty:** Hard
- **Focus:** CI/CD, infrastructure, reliability
- **Categories:** DevOps, infrastructure, reliability

### 12. **Security Specialist** (`security`)
- **Duration:** 45-60 minutes
- **Questions:** 5
- **Difficulty:** Hard
- **Focus:** Application security, vulnerabilities
- **Categories:** Security, vulnerabilities, secure coding

### 13. **Full Interview Day** (`full_interview`)
- **Duration:** 3 hours
- **Questions:** 18 (distributed across multiple rounds)
- **Difficulty:** Mixed
- **Focus:** Complete interview day simulation
- **Categories:** Mixed sequence (NOT all domains, but realistic round progression)

**How it works:**
The system simulates a **real full-day interview** with questions from different rounds in sequence:

**Junior (0-3 years) - 18 questions:**
```
1. HR Screening (3 questions) - "Tell me about yourself..."
2. Technical Round 1 (6 questions) - Core fundamentals
3. Behavioral (4 questions) - "Describe a time when..."
4. Technical Round 2 (5 questions) - Problem-solving
```

**Mid-Level (3-7 years) - 18 questions:**
```
1. HR Screening (2 questions)
2. Technical Round 1 (5 questions) - Fundamentals
3. Technical Round 2 (6 questions) - Advanced topics
4. Behavioral (5 questions) - Leadership scenarios
```

**Senior (7+ years) - 18 questions:**
```
1. HR Screening (2 questions)
2. Technical Round 2 (5 questions) - Deep technical
3. System Design (3 questions) - Architecture challenges
4. Behavioral (4 questions) - Impact stories
5. Managerial (4 questions) - Strategic thinking
```

**Example:** If you're a 5-year Python Backend engineer:
- Q1-2: HR questions about your background
- Q3-7: Technical fundamentals (Python, APIs, etc.)
- Q8-13: Advanced technical (architecture, scaling)
- Q14-18: Behavioral (teamwork, challenges)

---

## API Endpoints

### 1. Get Available Rounds

**GET** `/api/practice/rounds/available`

Get all available interview rounds with optional personalized recommendations.

#### Query Parameters (Optional)
```typescript
{
  experience_years?: number  // Years of experience (0-50)
  domain?: string            // Domain (e.g., "Python Backend", "Data Science")
}
```

#### Response
```json
{
  "rounds": [
    {
      "round_type": "hr_screening",
      "name": "HR Screening Round",
      "description": "Initial screening focusing on background, motivation, and culture fit",
      "duration_minutes": 20,
      "question_count": 4,
      "difficulty": "easy",
      "question_time_limit": 90,
      "categories": ["behavioral", "motivation", "background"]
    },
    ...
  ],
  "recommended_round": "technical_round_1",  // Based on experience
  "recommended_sequence": [                   // Suggested progression
    "hr_screening",
    "technical_round_1",
    "behavioral"
  ]
}
```

#### Example Requests
```bash
# Get all rounds
curl http://localhost:8000/api/practice/rounds/available

# Get rounds with recommendations for 2 years experience
curl http://localhost:8000/api/practice/rounds/available?experience_years=2

# Get rounds relevant to Data Engineering
curl http://localhost:8000/api/practice/rounds/available?domain=Data%20Engineering

# Get personalized recommendations
curl http://localhost:8000/api/practice/rounds/available?experience_years=5&domain=Python%20Backend
```

---

### 2. Start Round-Based Interview

**POST** `/api/practice/interview/start-round`

Start an interview session for a specific round.

#### Request Body
```typescript
{
  round_type: "technical_round_1" | "system_design" | ... ,  // REQUIRED: Interview round
  domain: string,                                             // REQUIRED: Primary domain (e.g., "Python", "Data Engineering")
  experience_years: number,                                   // REQUIRED: Years of experience (default: 2)
  company_specific?: string,                                  // OPTIONAL: e.g., "Google", "Meta", "Amazon"
  user_profile?: {                                            // OPTIONAL: Complete profile (overrides domain/experience)
    domain: string,
    experience_years: number,
    skills: string[],
    job_role?: string,
    company_preference?: string,
    interview_focus?: string[],
    target_round?: InterviewRound
  }
}
```

**⚠️ CRITICAL: `domain` is REQUIRED!**
Without a domain, the system cannot generate relevant questions for your interview round.

**Supported Domains:**
- **Backend:** `"Python"`, `"Java"`, `"Node.js"`, `"Go"`, `"C#"`
- **Frontend:** `"React"`, `"Angular"`, `"Vue"`, `"JavaScript"`, `"TypeScript"`
- **Data:** `"Data Engineering"`, `"Data Science"`, `"Machine Learning"`
- **DevOps:** `"DevOps"`, `"Cloud Engineering"`, `"SRE"`
- **Other:** `"Security Engineering"`, `"Mobile Development"`

#### Response
```json
{
  "session_id": "abc-123-def-456",
  "first_question": {
    "id": 1,
    "text": "Explain the difference between fact and dimension tables...",
    "difficulty": "medium",
    "time_limit": 120,
    "category": "technical",
    "key_points": ["fact tables", "dimension tables", "star schema"],
    "expected_answer_template": "Should cover definitions and relationships",
    "round_type": "technical_round_1"
  },
  "tts_audio_url": "/api/practice/audio/abc-123-def-456_q1.mp3",
  "total_questions": 6,
  "progress": "1/6"
}
```

#### Example Requests

**✅ Recommended: Simple with domain (Most common use case):**
```bash
curl -X POST http://localhost:8000/api/practice/interview/start-round \
  -H "Content-Type: application/json" \
  -d '{
    "round_type": "technical_round_1",
    "domain": "Python",
    "experience_years": 3
  }'
```

**✅ With company-specific customization:**
```bash
curl -X POST http://localhost:8000/api/practice/interview/start-round \
  -H "Content-Type: application/json" \
  -d '{
    "round_type": "system_design",
    "domain": "Data Engineering",
    "experience_years": 5,
    "company_specific": "Google"
  }'
```

**✅ Advanced: Complete user profile (overrides domain/experience):**
```bash
curl -X POST http://localhost:8000/api/practice/interview/start-round \
  -H "Content-Type: application/json" \
  -d '{
    "round_type": "technical_round_2",
    "domain": "Data Engineering",
    "experience_years": 5,
    "user_profile": {
      "domain": "Data Engineering",
      "experience_years": 5,
      "skills": ["Python", "Spark", "Kafka", "Airflow"],
      "job_role": "Senior Data Engineer"
    }
  }'
```

**Company-Specific:**
```bash
curl -X POST http://localhost:8000/api/practice/interview/start-round \
  -H "Content-Type: application/json" \
  -d '{
    "round_type": "behavioral",
    "company_specific": "Amazon",
    "user_profile": {
      "domain": "Backend Engineering",
      "experience_years": 3,
      "skills": ["Python", "AWS", "Docker"]
    }
  }'
```

---

### 3. Quick Start with Round Selection (Enhanced)

**POST** `/api/practice/interview/quick-start`

Now supports round selection via `target_round` parameter.

#### Request Body
```typescript
{
  voice_input?: string,
  context?: string,
  auto_mode?: boolean = true,
  session_memory?: boolean = true,
  question_count?: number,
  target_company?: string,
  target_round?: InterviewRound  // NEW - Select specific round
}
```

#### Example
```bash
curl -X POST http://localhost:8000/api/practice/interview/quick-start \
  -H "Content-Type: application/json" \
  -d '{
    "voice_input": "I'm preparing for Senior Backend Engineer role",
    "target_round": "system_design",
    "target_company": "Google"
  }'
```

---

## Frontend Integration

### UI Flow Example

```typescript
// 1. Fetch available rounds
const response = await fetch('/api/practice/rounds/available?experience_years=5');
const { rounds, recommended_round } = await response.json();

// 2. Display rounds to user
<select onChange={handleRoundSelection}>
  {rounds.map(round => (
    <option value={round.round_type}>
      {round.name} - {round.duration_minutes} min, {round.question_count} questions
    </option>
  ))}
</select>

// 3. Start selected round
const startRound = async (roundType) => {
  const response = await fetch('/api/practice/interview/start-round', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      round_type: roundType,
      user_profile: {
        domain: "Python Backend",
        experience_years: 5,
        skills: ["Python", "Django", "PostgreSQL", "AWS"]
      }
    })
  });
  
  const { session_id, first_question, tts_audio_url } = await response.json();
  // Start interview...
};
```

### Round Selection Component

```jsx
function RoundSelector({ onSelectRound }) {
  const [rounds, setRounds] = useState([]);
  const [recommended, setRecommended] = useState(null);
  
  useEffect(() => {
    fetch('/api/practice/rounds/available?experience_years=5&domain=Python Backend')
      .then(res => res.json())
      .then(data => {
        setRounds(data.rounds);
        setRecommended(data.recommended_round);
      });
  }, []);
  
  return (
    <div className="round-selector">
      <h2>Select Interview Round</h2>
      
      {recommended && (
        <div className="recommended-badge">
          Recommended for you: {roundsMap[recommended].name}
        </div>
      )}
      
      <div className="round-grid">
        {rounds.map(round => (
          <div 
            key={round.round_type}
            className={`round-card ${round.round_type === recommended ? 'recommended' : ''}`}
            onClick={() => onSelectRound(round.round_type)}
          >
            <h3>{round.name}</h3>
            <div className="round-info">
              <span>⏱️ {round.duration_minutes} min</span>
              <span>❓ {round.question_count} questions</span>
              <span className={`difficulty-${round.difficulty}`}>
                {round.difficulty.toUpperCase()}
              </span>
            </div>
            <p>{round.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## Round Sequences (Recommended Progressions)

### Junior (0-3 years)
```
1. HR Screening
2. Technical Round 1
3. Behavioral
```

### Mid-Level (3-7 years)
```
1. HR Screening
2. Technical Round 1
3. Technical Round 2
4. Behavioral
```

### Senior (7+ years)
```
1. HR Screening
2. Technical Round 2
3. System Design
4. Behavioral
5. Managerial (optional)
```

---

## Benefits

✅ **Targeted Practice:** Focus on specific weaknesses  
✅ **Realistic Experience:** Matches real interview processes  
✅ **Time-Efficient:** Practice only what you need  
✅ **Progressive Learning:** Follow recommended sequences  
✅ **Company-Specific:** Adapt to target company styles  
✅ **Mixed Round Simulation:** Full interview day mimics real company processes (HR → Technical → Behavioral → etc.)

---

## Important: Full Interview Day Explained

**Question: "Does 'Full Interview' include all domain rounds (ML + Data Engineering + Frontend)?"**

**Answer: NO!** ❌

The Full Interview Day simulates a **realistic single-role interview**, not a mix of unrelated domains.

**What it actually does:**
- Uses **your profile** (e.g., Python Backend, 5 years)
- Generates a **realistic interview sequence** for THAT role
- Follows **actual company interview processes**

**Example for "Python Backend, 5 years experience":**
```
Round 1: HR Screening (2 questions)
  → "Tell me about your backend experience"
  → "Why are you interested in this role?"

Round 2: Technical Round 1 (5 questions)
  → Python fundamentals
  → API design questions
  → Database queries

Round 3: Technical Round 2 (6 questions)
  → Microservices architecture
  → Performance optimization
  → Async programming

Round 4: Behavioral (5 questions)
  → "Describe a time you improved system performance"
  → "How do you handle technical disagreements?"
```

**It does NOT give you:**
- ❌ ML questions if you're a Backend engineer
- ❌ Frontend questions if you're a Data Engineer
- ❌ Random mix of unrelated topics

**It DOES give you:**
- ✅ Progressive difficulty within YOUR domain
- ✅ Multiple rounds like real interviews (HR → Tech → Behavioral)
- ✅ Realistic flow (screening → technical depth → soft skills)
- ✅ Experience-level appropriate (Junior vs Senior sequences differ)

---

## Use Cases

### 1. **Weak in System Design**
```
User selects: "System Design Round"
→ Gets 2-3 deep system design questions
→ Practices scalability, tradeoffs, architecture
```

### 2. **Preparing for Google L5**
```
User selects: Technical Round 2 + "Google"
→ Gets Google-style technical questions
→ Senior-level difficulty
```

### 3. **First Job (Junior)**
```
System recommends: HR Screening → Technical Round 1
→ Appropriate difficulty for beginners
→ Builds confidence gradually
```

### 4. **Full Interview Simulation**
```
User selects: "Full Interview Day"
→ 20 questions across all rounds
→ 3-4 hour complete experience
```

---

## Testing

```bash
# Test round listing
curl http://localhost:8000/api/practice/rounds/available

# Test system design round
curl -X POST http://localhost:8000/api/practice/interview/start-round \
  -H "Content-Type: application/json" \
  -d '{"round_type": "system_design"}'

# Test with profile
curl -X POST http://localhost:8000/api/practice/interview/start-round \
  -H "Content-Type: application/json" \
  -d '{
    "round_type": "machine_learning",
    "user_profile": {
      "domain": "Machine Learning",
      "experience_years": 4,
      "skills": ["Python", "TensorFlow", "PyTorch", "MLOps"]
    }
  }'
```

---

## Questions?

The round-based system integrates seamlessly with:
- ✅ Existing feedback system (correctness scoring)
- ✅ Speech analytics
- ✅ TTS/STT services
- ✅ Adaptive question generation
- ✅ Company-specific customization

All existing features work with round-based interviews!
