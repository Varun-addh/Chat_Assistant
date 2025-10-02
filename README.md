# Interview Assistant Backend

FastAPI backend for a real-time interview assistant. Supports REST for Q&A and WebSocket for audio streaming and partial transcripts.

## Features
- Session creation and per-session Q&A history
- AI answer generation (short or detailed) via Groq (or mock without API key)
- WebSocket audio ingestion with streaming partial transcripts (pluggable STT)
- Optional JSONL audit logging of Q&A
- Simple API key authentication

## Requirements
- Python 3.11+

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-70b-versatile
ANSWER_TEMPERATURE=0.4
API_KEY=dev-secret
CORS_ALLOW_ORIGINS=*
STT_PROVIDER=none
ANALYTICS_PATH=logs/qna.jsonl
```

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API
- POST `/api/session` → `{ session_id }`
- POST `/api/question` → `{ answer, style, created_at }`
- GET `/api/history/{session_id}` → session history
- WS `/ws/stt/{session_id}`: send binary audio frames, optional text `__end__` to close. Receives `{ type: "partial_transcript", text }` events.

Authentication:
- REST: set header `Authorization: Bearer <API_KEY>` if configured
- WS: set `Sec-WebSocket-Protocol: <API_KEY>` if configured

## Notes
- When no `GROQ_API_KEY` is set, answers are mocked and STT yields placeholders.
- Plug actual STT in `app/services/stt_service.py`.
