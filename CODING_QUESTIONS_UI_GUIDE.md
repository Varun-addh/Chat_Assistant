# Coding Questions UI Implementation Guide

## Problem
AI generates **coding questions** (e.g., "Write Python code to...") but UI shows **voice recorder** instead of **code editor**.

## Solution
Backend now sets `question_type` field to indicate how the question should be answered.

---

## Backend Changes (✅ COMPLETE)

### 1. New `question_type` Field
```typescript
type QuestionType = "voice" | "coding" | "system_design";
```

### 2. Question Schema
```typescript
interface PracticeInterviewQuestion {
  id: number;
  text: string;
  difficulty: "easy" | "medium" | "hard";
  time_limit: number;  // Seconds
  category: string;
  
  // NEW FIELDS
  question_type: QuestionType;  // Determines UI
  programming_language?: string;  // For coding questions: "Python", "JavaScript", "SQL"
  code_template?: string;  // Starter code
  test_cases?: Array<{input: any, expected_output: any}>;
  
  key_points?: string[];
  expected_answer_template?: string;
  round_type?: string;
}
```

### 3. Time Limits by Type
- **Voice questions:** 60-180s (1-3 minutes)
- **Coding questions:** 600-900s (10-15 minutes) ← AI automatically sets this
- **System design:** 900-1800s (15-30 minutes)

### 4. AI Auto-Detection
The AI now automatically detects coding questions by looking for:
- "Write the code"
- "Write a function"
- "Implement"
- "Write Python/JavaScript/SQL"
- "Code snippet"
- "Create a function"

When detected, sets:
```json
{
  "question_type": "coding",
  "programming_language": "Python",
  "time_limit": 600
}
```

---

## Frontend Implementation Required

### Step 1: Check Question Type
```typescript
const renderQuestionUI = (question: PracticeInterviewQuestion) => {
  switch (question.question_type) {
    case "coding":
      return <CodeEditor question={question} />;
    
    case "system_design":
      return <WhiteboardEditor question={question} />;
    
    case "voice":
    default:
      return <VoiceRecorder question={question} />;
  }
};
```

### Step 2: Code Editor Component
```typescript
import MonacoEditor from '@monaco-editor/react';

const CodeEditor = ({ question }: { question: PracticeInterviewQuestion }) => {
  const [code, setCode] = useState(question.code_template || '');
  const [timeRemaining, setTimeRemaining] = useState(question.time_limit);

  const handleSubmit = async () => {
    const response = await fetch('/api/practice/interview/submit-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: currentSessionId,
        questionId: question.id,
        code: code,
        programmingLanguage: question.programming_language,
        timeTaken: question.time_limit - timeRemaining
      })
    });

    const result = await response.json();
    // Show test results and feedback
    showCodeFeedback(result);
  };

  return (
    <div className="code-editor-container">
      <div className="question-header">
        <h3>{question.text}</h3>
        <div className="timer">{formatTime(timeRemaining)}</div>
      </div>

      <MonacoEditor
        height="400px"
        language={question.programming_language?.toLowerCase() || 'python'}
        value={code}
        onChange={(value) => setCode(value || '')}
        theme="vs-dark"
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
        }}
      />

      <div className="editor-actions">
        <button onClick={handleSubmit} className="submit-code-btn">
          Submit Code
        </button>
        <button onClick={runTests} className="run-tests-btn">
          Run Tests
        </button>
      </div>
    </div>
  );
};
```

### Step 3: Install Monaco Editor
```bash
npm install @monaco-editor/react
```

### Step 4: Code Feedback Display
```typescript
const CodeFeedbackPanel = ({ result }: { result: SubmitCodeResponse }) => {
  return (
    <div className="code-feedback">
      <h3>Test Results</h3>
      
      {result.test_results.map((test, idx) => (
        <div key={idx} className={`test-case ${test.passed ? 'passed' : 'failed'}`}>
          <div className="test-header">
            Test {test.test_case_id}: {test.passed ? '✅ Passed' : '❌ Failed'}
          </div>
          {!test.passed && (
            <div className="test-details">
              <div>Input: {test.input_data}</div>
              <div>Expected: {test.expected_output}</div>
              <div>Got: {test.actual_output}</div>
              {test.error && <div className="error">{test.error}</div>}
            </div>
          )}
        </div>
      ))}

      <div className="code-quality">
        <h4>Code Quality Feedback</h4>
        <div className="score">
          Correctness: {result.code_feedback.correctness_score}%
        </div>
        <div className="approach">
          Approach: {result.code_feedback.approach_quality}
        </div>
        
        {result.code_feedback.time_complexity && (
          <div>Time Complexity: {result.code_feedback.time_complexity}</div>
        )}
        
        {result.code_feedback.strengths.length > 0 && (
          <div className="strengths">
            <h5>Strengths:</h5>
            <ul>
              {result.code_feedback.strengths.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
        )}
        
        {result.code_feedback.improvements.length > 0 && (
          <div className="improvements">
            <h5>Improvements:</h5>
            <ul>
              {result.code_feedback.improvements.map((i, idx) => <li key={idx}>{i}</li>)}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};
```

---

## Example API Responses

### Voice Question (Current Behavior)
```json
{
  "id": 1,
  "text": "Tell me about a time you resolved a conflict in your team",
  "difficulty": "medium",
  "time_limit": 90,
  "category": "behavioral",
  "question_type": "voice"
}
```
→ UI shows **microphone button**

### Coding Question (NEW)
```json
{
  "id": 2,
  "text": "Write the Python code snippet to calculate the total purchase amount for the top 5 customers",
  "difficulty": "easy",
  "time_limit": 600,
  "category": "technical",
  "question_type": "coding",
  "programming_language": "Python",
  "key_points": ["groupby", "sum", "nlargest/head", "pandas"],
  "expected_answer_template": "Use pandas groupby to aggregate by customer, then sort and take top 5"
}
```
→ UI shows **code editor with 10-minute timer**

### System Design Question (Future)
```json
{
  "id": 3,
  "text": "Design a URL shortener system like bit.ly",
  "difficulty": "hard",
  "time_limit": 1800,
  "category": "system_design",
  "question_type": "system_design"
}
```
→ UI shows **whiteboard/drawing tool**

---

## Testing

1. **Start a new interview** with technical round
2. **Check network tab** for question response
3. **Verify `question_type` field** is present
4. **If coding question:**
   - `question_type === "coding"` ✅
   - `time_limit >= 600` ✅
   - `programming_language` is set ✅
5. **UI should show code editor** instead of microphone

---

## Quick Win: Minimal Implementation

If you don't have time for Monaco Editor, at minimum:

```typescript
{question.question_type === "coding" ? (
  <div>
    <h3>Coding Question</h3>
    <p>{question.text}</p>
    <textarea
      rows={15}
      value={code}
      onChange={(e) => setCode(e.target.value)}
      placeholder="Write your code here..."
      style={{ fontFamily: 'monospace', width: '100%' }}
    />
    <button onClick={submitCode}>Submit Code</button>
  </div>
) : (
  <VoiceRecorder question={question} />
)}
```

This gives you a basic code editor while you implement Monaco later!

---

## Summary

✅ **Backend is ready** - AI detects coding questions and sets `question_type: "coding"`
✅ **Time limits adjusted** - Coding questions get 10-15 minutes automatically
⏳ **Frontend needs update** - Check `question_type` and render code editor

**Next step:** Update frontend `PracticeMode.tsx` to check `question.question_type` and render conditionally! 🎯
