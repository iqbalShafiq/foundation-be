# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

**Run the application:**
```bash
python main.py
```

**Setup environment (when needed):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Then edit .env with OPENAI_API_KEY and SECRET_KEY
```

**Testing:**
Note: This project references test files in the README, but they don't currently exist in the repository. Any testing would need to be implemented first.

## Architecture

This is a FastAPI chatbot application with streaming support using LangChain and OpenAI.

**Core Structure:**
- `main.py` - FastAPI app with CORS middleware, exception handling, and database setup
- `app/models.py` - Pydantic and SQLAlchemy models including User, roles, and chat models
- `app/database.py` - SQLite database configuration and session management  
- `app/services/` - Business logic services (chat_service.py, auth_service.py, conversation_service.py)
- `app/dependencies.py` - FastAPI dependencies for authentication and authorization
- `app/routers/` - API route handlers (thin layer, delegates to services)

**Key Architecture Details:**
- JWT-based authentication with role-based access control (admin, user)
- SQLite database with SQLAlchemy ORM for user management and automatic table creation on startup
- Password hashing using bcrypt via passlib for security
- ChatService singleton with lazy initialization of ChatOpenAI clients per model type
- Model mapping: Fast→gpt-4.1-mini, Standard→gpt-4.1, Fast Reasoning→o4-mini, Reasoning→o3
- **Message Branching System**: Supports editing messages with branch creation - when a user message is edited, a new branch is created and AI responses are regenerated while preserving original conversation history
- Streaming responses use Server-Sent Events format with JSON data chunks
- CORS middleware enabled for all origins with full credentials support
- Custom exception handlers for validation errors and HTTP exceptions with user-friendly error mapping
- Environment variables managed via python-dotenv with .env.example template

**API Endpoints:**

**Authentication:**
- `POST /auth/register` - Register new user (username, email, password, role)
- `POST /auth/login` - Login and get JWT token
- `GET /auth/me` - Get current user info (requires auth)
- `GET /auth/users` - List all users (admin only)
- `DELETE /auth/users/{user_id}` - Delete user (admin only)

**Chat:**
- `POST /chat` - Chat with streaming responses (requires auth)
- `GET /conversations` - Get user conversations with optional keyword search and related chats (requires auth)
- `GET /conversations/{conversation_id}` - Get detailed conversation with message history (requires auth)

**Messages (Message Editing & Branching):**
- `PUT /messages/{message_id}` - Edit a user message and create new branch with regenerated AI responses (requires auth)
- `GET /conversations/{conversation_id}/branches` - Get all branches for a conversation (requires auth)
- `POST /conversations/{conversation_id}/branches/{branch_id}/activate` - Switch active branch for a conversation (requires auth)

**Feedback:**
- `POST /feedback` - Add feedback (like/dislike) with optional description for a message (requires auth)
- `GET /feedback/message/{message_id}` - Get all feedback for a specific message (requires auth)
- `DELETE /feedback/{feedback_id}` - Delete user's own feedback (requires auth)

**System:**
- `GET /health` - Health check
- `GET /docs` - FastAPI auto-generated documentation

**Authentication Usage:**
1. Register: `POST /auth/register` with `{"username": "user", "email": "user@example.com", "password": "pass", "role": "user"}`
2. Login: `POST /auth/login` with `{"username": "user", "password": "pass"}` → returns JWT token
3. Use token: Add `Authorization: Bearer <token>` header to protected endpoints

## Architecture Rules & Best Practices

**IMPORTANT: Follow these architectural principles when creating new endpoints or modifying existing ones:**

### Service Layer Pattern
1. **Router Layer (Thin)**: Handle HTTP concerns only
   - Parameter validation (FastAPI automatic)
   - Authentication/authorization (via dependencies)
   - HTTP status codes and responses
   - Delegate business logic to services
   - Keep router methods under 10 lines when possible

2. **Service Layer (Business Logic)**: Handle all business logic
   - Database operations and queries
   - Data processing and transformations
   - Business rules and validations
   - Complex logic and algorithms
   - Return domain objects or simple data structures

3. **Model Layer**: Data structures and database models
   - SQLAlchemy models for database tables
   - Pydantic models for API request/response validation
   - Keep models focused on data structure only

### Code Organization Rules

**When creating new endpoints:**
1. Create a service class in `app/services/` for business logic
2. Keep routers thin - they should only handle HTTP concerns
3. Use dependency injection for database sessions
4. Follow existing naming conventions

**Example Structure:**
```python
# Router (thin layer)
@router.get("/endpoint")
async def get_data(db: Session = Depends(get_db)):
    service = MyService(db)
    return service.get_data()

# Service (business logic)
class MyService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_data(self):
        # Complex business logic here
        pass
```

**Service Class Guidelines:**
- One service per domain/feature area
- Use dependency injection for database sessions
- Break complex methods into smaller private methods
- Use descriptive method names that explain business intent
- Handle error cases and return appropriate responses

**Benefits of this pattern:**
- Better testability (services can be unit tested independently)
- Cleaner separation of concerns
- Reusable business logic
- Easier maintenance and debugging
- Better code organization and readability