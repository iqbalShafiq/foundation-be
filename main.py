from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.routers import chat, health, auth, feedback, preferences, gallery, documents, messages
from app.database import engine
from app.models import Base

load_dotenv()

app = FastAPI(title="Chatbot API", version="1.0.0")

Base.metadata.create_all(bind=engine)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Handle validation errors with a generic response format"""
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid request data",
            "message": "Please check your input parameters",
            "details": exc.errors()  # Include details for debugging
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPException with custom error format"""
    # Map common error messages to user-friendly ones
    error_mapping = {
        "Incorrect username or password": {
            "error": "Authentication failed",
            "message": "Invalid username or password"
        },
        "Could not validate credentials": {
            "error": "Authentication failed", 
            "message": "Invalid or expired token"
        },
        "Username already registered": {
            "error": "Registration failed",
            "message": "Username is already taken"
        },
        "Email already registered": {
            "error": "Registration failed",
            "message": "Email is already registered"
        },
        "User not found": {
            "error": "User not found",
            "message": "The requested user does not exist"
        },
        "Not enough permissions": {
            "error": "Access denied",
            "message": "You don't have permission to perform this action"
        },
        "Inactive user": {
            "error": "Account inactive",
            "message": "Your account has been deactivated"
        }
    }
    
    # Use custom mapping if available, otherwise use generic format
    if exc.detail in error_mapping:
        content = error_mapping[exc.detail]
    else:
        content = {
            "error": "Request failed",
            "message": str(exc.detail)
        }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=getattr(exc, 'headers', None)
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(messages.router)
app.include_router(documents.router)
app.include_router(gallery.router)
app.include_router(feedback.router)
app.include_router(preferences.router)
app.include_router(health.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
