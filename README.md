# Chatbot API

A modular FastAPI application for a streaming chatbot using LangChain and OpenAI.

## Project Structure

```
foundation-be/
├── app/
│   ├── __init__.py
│   ├── models.py           # Pydantic models
│   ├── routers/            # API route handlers
│   │   ├── __init__.py
│   │   ├── chat.py         # Chat endpoints
│   │   └── health.py       # Health check endpoint
│   └── services/           # Business logic
│       ├── __init__.py
│       └── chat_service.py # Chat service with LangChain integration
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_client.py      # Integration tests for API endpoints
│   └── test_structure.py   # Module structure validation
├── main.py                 # FastAPI app initialization
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## Testing

- **Test API endpoints:**
  ```bash
  python tests/test_client.py
  ```

- **Test module structure:**
  ```bash
  python tests/test_structure.py
  ```

## API Endpoints

- `POST /chat` - Chat with streaming or non-streaming responses
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation

## Features

- **Modular Architecture:** Clean separation of concerns with routers, services, and models
- **Streaming Support:** Real-time streaming responses using Server-Sent Events
- **CORS Enabled:** Cross-origin requests supported
- **Type Safety:** Full Pydantic model validation
- **Lazy Loading:** ChatOpenAI client initialized only when needed