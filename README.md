---
title: Stratax AI - Interview Assistant
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Stratax AI - Interview Intelligence Platform

An advanced AI-powered interview preparation platform featuring:

- 🤖 **AI Copilot Chat Assistant** - Get expert answers to technical and behavioral interview questions
- 📊 **Multi-View System Design** - Generate comprehensive architecture diagrams with 5 focused views
- 🎤 **Mock Interview Practice** - Practice with AI interviewer and get real-time feedback
- 🔍 **Search Intelligence** - Find curated interview questions from top companies
- 📈 **Live Practice Mode** - Real-time speech-to-text practice with analytics

## Features

### 1. AI Copilot Chat
- Context-aware responses using advanced LLMs (Gemini/Groq)
- Intelligent API provider selection based on configured keys
- System design architecture generation (single-view and multi-view)
- Code evaluation and explanation

### 2. Interview Intelligence
- Curated questions from FAANG+ companies
- Difficulty-based filtering (Easy, Medium, Hard)
- Topic-specific question generation
- Company-specific interview prep

### 3. Mock Interview
- Simulated interview sessions
- Real-time feedback and scoring
- Adaptive difficulty based on performance
- Detailed performance analytics

### 4. Live Practice Mode
- Speech-to-text transcription
- Real-time answer evaluation
- Speaking pace and clarity analysis
- Offline TTS for interviewer questions

## Technology Stack

- **Backend**: FastAPI (Python 3.12)
- **AI/LLM**: Google Gemini API, Groq API
- **Vector DB**: Qdrant (for semantic search)
- **Embeddings**: SentenceTransformers
- **Architecture**: Microservices with Docker

## Configuration

The application supports flexible API key configuration:

- **Only Groq API key**: All features use Groq
- **Only Gemini API key**: All features use Gemini
- **Both API keys**: AI Copilot uses Gemini, other features use Groq

Set your API keys in the Bridge Settings or via environment variables.

## Development

See [README_DOCKER.md](README_DOCKER.md) for Docker deployment instructions.

## License

MIT License - See LICENSE file for details

---

**Developed by Varun Bikkumalla**
