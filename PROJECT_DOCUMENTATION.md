# InterviewAst - AI-Powered Interview Assistant

## Executive Summary

InterviewAst is a comprehensive AI-powered interview preparation platform designed to help candidates excel in technical and behavioral interviews. The system provides real-time assistance through structured Q&A sessions, personalized feedback, and comprehensive preparation materials.

## Project Overview

### Purpose
To create an intelligent interview assistant that helps candidates prepare for technical and behavioral interviews by providing:
- Structured, interview-ready responses
- Personalized feedback based on candidate profiles
- Real-time conversation history and context
- Multiple response styles and tones

### Key Features
- **Session Management**: Persistent conversation history with unique session IDs
- **AI-Powered Responses**: Integration with Groq/Gemini LLMs for intelligent answers
- **Contextual Awareness**: Maintains conversation context and candidate profiles
- **Flexible Response Styles**: Multiple tones (Mentor, Evaluator, Peer) and formats
- **Real-time Streaming**: WebSocket support for live response streaming
- **Profile Integration**: Upload and utilize candidate resumes/profiles for personalized responses
- **Smart Query Handling**: Automatic detection and handling of ambiguous, off-topic, and greeting queries
- **Defensive Programming**: Built-in guidelines for robust code examples and error handling
- **Context Fallback**: Intelligent fallback when conversation context is insufficient
- **Token Management**: Adaptive token limits based on question complexity
- **Visual Support**: Text-based descriptions for architecture diagrams and flowcharts
- **Citation Guidelines**: Proper handling of external sources and knowledge disclaimers

## Technical Architecture

### Technology Stack
- **Backend**: FastAPI (Python 3.11+)
- **AI/LLM**: Groq API, Google Gemini API
- **Authentication**: API Key-based security
- **Storage**: JSON file-based session persistence
- **Real-time**: WebSocket for streaming responses
- **Audio**: Pluggable STT (Speech-to-Text) service

### System Components

#### 1. Core Services
- **LLMService**: Handles AI response generation and formatting
- **SessionManager**: Manages conversation sessions and persistence
- **STTService**: Speech-to-text processing (pluggable)
- **AuditService**: Logging and analytics

#### 2. API Endpoints
- **Session Management**: Create, list, delete sessions
- **Q&A Processing**: Submit questions, receive structured answers
- **Profile Management**: Upload and manage candidate profiles
- **History Management**: View and manage conversation history

#### 3. Data Models
- **SessionState**: Session data structure with Q&A history
- **QuestionIn/AnswerOut**: Request/response schemas
- **Profile Integration**: Resume and background information

## Key Features & Capabilities

### 1. Intelligent Response Generation
- **Structured Format**: Consistent heading structure with bold formatting
- **Context Awareness**: Maintains conversation context across sessions
- **Style Adaptation**: Multiple response styles (concise, deep-dive, mentor, executive)
- **Greeting Handling**: Special handling for salutations and casual interactions
- **Ambiguous Query Detection**: Automatically identifies unclear questions and asks for clarification
- **Off-Topic Redirect**: Politely redirects non-interview questions to relevant topics
- **Context Fallback**: Provides standalone answers when conversation context is insufficient
- **Token Management**: Adaptive response length based on question complexity (300-1200 tokens)

### 2. Session Management
- **Persistent Storage**: JSON-based session persistence
- **History Tracking**: Complete Q&A history with timestamps
- **Profile Integration**: Candidate-specific personalized responses
- **Session Lifecycle**: Create, update, delete, and clear sessions

### 3. Response Customization
- **Tone Selection**: Mentor, Evaluator, Peer, Executive, Academic, Coaching
- **Layout Options**: Bullets, Narrative, Q&A, FAQ, Checklist, Pros-Cons
- **Style Modes**: Auto, Varied, Concise, Deep-dive, Mentor, Executive
- **Variability Control**: Configurable response variation (0-1 scale)

### 4. Technical Interview Support
- **Code Examples**: Syntax-highlighted code blocks with defensive programming
- **Algorithm Analysis**: Time/space complexity explanations
- **System Design**: Architecture and component discussions
- **Best Practices**: Industry-standard approaches and patterns
- **Visual Descriptions**: Text-based architecture diagrams and flowcharts
- **Error Handling**: Comprehensive error handling patterns and edge cases
- **Input Validation**: Defensive coding practices and boundary condition handling

## API Documentation

### Core Endpoints

#### Session Management
```
POST /api/session
- Creates new interview session
- Returns: { session_id }

GET /api/sessions
- Lists all sessions with metadata
- Returns: { items: [{ session_id, last_update, qna_count }] }

DELETE /api/session/{session_id}
- Deletes entire session and data
- Returns: { status: "ok", deleted: true }
```

#### Question & Answer
```
POST /api/question
- Submits question for AI processing
- Parameters: session_id, question, optional style settings
- Returns: { answer, created_at }
- Supports streaming via stream=true parameter
```

#### Profile Management
```
POST /api/upload_profile
- Uploads candidate profile/resume
- Parameters: file, session_id
- Enables personalized responses
```

#### History Management
```
GET /api/history/{session_id}
- Retrieves complete conversation history
- Returns: { qna: [{ question, answer, created_at }] }

DELETE /api/history/{session_id}
- Clears Q&A history (keeps session)
- Returns: { status: "ok" }
```

### WebSocket Support
```
WS /ws/stt/{session_id}
- Real-time audio processing
- Sends: Binary audio frames
- Receives: { type: "partial_transcript", text }
```

## Configuration & Setup

### Environment Variables
```bash
# AI Provider Configuration
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-70b-versatile
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-pro

# Application Settings
API_KEY=dev-secret
ANSWER_TEMPERATURE=0.4
CORS_ALLOW_ORIGINS=*

# Optional Features
STT_PROVIDER=none
ANALYTICS_PATH=logs/qna.jsonl
```

### Installation & Deployment
```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Data Storage & Persistence

### Session Storage
- **Format**: JSON files in `data/sessions/`
- **Structure**: Session metadata, Q&A history, profile data
- **Persistence**: Survives server restarts
- **Management**: Manual deletion via API endpoints

### Data Models
```json
{
  "session_id": "uuid",
  "qna": [
    {
      "question": "string",
      "answer": "string", 
      "created_at": "ISO timestamp"
    }
  ],
  "profile_text": "string",
  "last_update": "ISO timestamp"
}
```

## Security & Authentication

### API Security
- **API Key Authentication**: Bearer token required for all endpoints
- **CORS Configuration**: Configurable allowed origins
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Comprehensive error responses

### Data Security
- **Session Isolation**: Each session is independently managed
- **No Sensitive Data**: No storage of personal information beyond profiles
- **Audit Logging**: Optional Q&A logging for analytics

## Performance & Scalability

### Current Performance
- **Response Time**: Sub-second for most queries
- **Concurrent Sessions**: Limited by server resources
- **Storage Growth**: Linear with session count
- **Memory Usage**: All sessions loaded in memory

### Scalability Considerations
- **Database Migration**: Consider PostgreSQL for production
- **Session Cleanup**: Implement automatic expiration
- **Caching**: Add Redis for session caching
- **Load Balancing**: Multiple server instances

## Recent Updates & Improvements

### Latest Features
1. **Heading Formatting**: Automatic bold formatting for all headings
2. **Greeting Detection**: Special handling for casual interactions
3. **Response Structure**: Consistent formatting with "Complete Answer" removal
4. **Session Persistence**: Enhanced session management and history tracking
5. **Smart Query Handling**: Automatic detection of ambiguous and off-topic queries
6. **Defensive Programming**: Built-in guidelines for robust code examples
7. **Context Fallback**: Intelligent handling when conversation context is insufficient
8. **Token Management**: Adaptive response length based on question complexity
9. **Visual Support**: Text-based descriptions for architecture diagrams
10. **Citation Guidelines**: Proper handling of external sources and disclaimers

### Code Quality
- **Type Hints**: Full Python type annotation
- **Error Handling**: Comprehensive exception management
- **Documentation**: Inline code documentation
- **Testing**: Unit tests for core functionality

## Future Roadmap

### Short-term Goals
- **Session Expiration**: Automatic cleanup of old sessions
- **Enhanced STT**: Better speech-to-text integration
- **Response Caching**: Improve response times
- **Analytics Dashboard**: Usage statistics and insights
- **Enhanced Query Detection**: Improved accuracy for ambiguous and off-topic queries
- **Visual Diagram Generation**: Support for actual diagram creation
- **Advanced Context Management**: Better conversation context understanding

### Long-term Vision
- **Multi-language Support**: International interview preparation
- **Video Integration**: Video-based interview practice
- **Company-specific Training**: Tailored content for specific companies
- **Mobile Application**: Native mobile app development

## Team & Development

### Development Process
- **Version Control**: Git-based workflow
- **Code Review**: Peer review process
- **Testing**: Automated testing pipeline
- **Documentation**: Comprehensive API documentation

### Maintenance
- **Regular Updates**: Dependency updates and security patches
- **Performance Monitoring**: Response time and error tracking
- **User Feedback**: Continuous improvement based on usage patterns

## Conclusion

InterviewAst represents a significant advancement in interview preparation technology, providing candidates with intelligent, personalized assistance for technical and behavioral interviews. The system's modular architecture, comprehensive API, and flexible configuration make it suitable for various deployment scenarios and future enhancements.

The platform successfully combines modern AI capabilities with practical interview preparation needs, offering a robust foundation for continued development and expansion.

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Prepared by**: Development Team  
**Status**: Production Ready
