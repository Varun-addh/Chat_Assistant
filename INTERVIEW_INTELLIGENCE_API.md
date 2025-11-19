# Interview Intelligence API Endpoints

**Base URL:** `http://127.0.0.1:8000/api/intelligence`

---

## 1. Get All Topics

**GET** `/topics`

Returns list of all available interview question topics.

**Response:**
```json
{
  "topics": ["aws", "data-science", "javascript", "python", "sql", "system-design"]
}
```

---

## 2. Get Questions by Topic

**GET** `/questions/{topic}?limit={limit}`

Get interview questions for a specific topic.

**Parameters:**
- `topic` (path, required): Topic name (e.g., "javascript", "data-science")
- `limit` (query, optional): Number of questions (1-100, default: 50)

**Example:**
```
GET /questions/javascript?limit=10
```

**Response:**
```json
{
  "topic": "javascript",
  "questions": [
    {
      "question": "What is the difference between let and var?",
      "answer": "The main differences are...",
      "source": "model_papers",
      "updated_at": "2025-11-06T07:32:48.828033"
    }
  ],
  "count": 10
}
```

---

## 3. Search Questions

**GET** `/search?q={query}&limit={limit}`

Intelligently search for interview questions across all topics.

**Parameters:**
- `q` (query, required): Search query (natural language)
- `limit` (query, optional): Number of results (1-50, default: 20)

**Example:**
```
GET /search?q=javascript%20closures&limit=5
```

**Response:**
```json
{
  "query": "javascript closures",
  "questions": [
    {
      "question": "Explain closures in JavaScript",
      "answer": "A closure is...",
      "source": "model_papers",
      "updated_at": "2025-11-06T07:32:48.828033",
      "topic": "javascript"
    }
  ],
  "count": 5
}
```

---

## 4. Trigger Update

**POST** `/update`

Manually trigger web scraping to update the question database.

**Response:**
```json
{
  "status": "ok",
  "message": "Update initiated. Questions will be available shortly."
}
```

**Note:** This is asynchronous. Update runs in background and may take several minutes.

---

## Response Schema

### InterviewQuestion
```typescript
{
  question: string;      // The interview question
  answer: string;        // Comprehensive answer
  source: string;        // Source URL or identifier
  updated_at: string;    // ISO timestamp
  topic?: string;        // Topic (only in search results)
}
```

---

## Quick Integration Examples

### JavaScript Fetch
```javascript
// Get topics
fetch('http://127.0.0.1:8000/api/intelligence/topics')
  .then(res => res.json())
  .then(data => console.log(data.topics));

// Get questions by topic
fetch('http://127.0.0.1:8000/api/intelligence/questions/javascript?limit=10')
  .then(res => res.json())
  .then(data => console.log(data.questions));

// Search questions
fetch('http://127.0.0.1:8000/api/intelligence/search?q=javascript&limit=5')
  .then(res => res.json())
  .then(data => console.log(data.questions));

// Trigger update
fetch('http://127.0.0.1:8000/api/intelligence/update', { method: 'POST' })
  .then(res => res.json())
  .then(data => console.log(data));
```

### Axios
```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000/api/intelligence'
});

// Get topics
api.get('/topics').then(res => console.log(res.data.topics));

// Get questions by topic
api.get('/questions/javascript', { params: { limit: 10 } })
  .then(res => console.log(res.data.questions));

// Search questions
api.get('/search', { params: { q: 'javascript', limit: 5 } })
  .then(res => console.log(res.data.questions));

// Trigger update
api.post('/update').then(res => console.log(res.data));
```

---

## Error Responses

All endpoints may return:
```json
{
  "detail": "Error message"
}
```

**Status Codes:**
- `200` - Success
- `400` - Bad Request
- `500` - Internal Server Error

