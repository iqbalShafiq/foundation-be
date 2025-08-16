from pydantic import BaseModel, EmailStr
from typing import Optional, List, Union
import uuid
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base


class ModelType(str, Enum):
    FAST = "Fast"
    STANDARD = "Standard"
    FAST_REASONING = "Fast Reasoning"
    REASONING = "Reasoning"


class ImageData(BaseModel):
    data: str  # Base64 encoded image data
    mime_type: str  # e.g., "image/jpeg", "image/png"
    url: Optional[str] = None  # URL to access the saved image


class ChatRequest(BaseModel):
    message: str
    model: ModelType = ModelType.STANDARD
    conversation_id: Optional[str] = None
    images: Optional[List[ImageData]] = None


class HealthResponse(BaseModel):
    status: str


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"


class FeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default=UserRole.USER.value)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.USER


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool


    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class UserUpdate(BaseModel):
    role: UserRole | None = None
    password: str | None = None


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    image_urls = Column(Text, nullable=True)  # JSON array of image URLs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    conversation = relationship("Conversation")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    feedback_type = Column(String, nullable=False)  # "like" or "dislike"
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    message = relationship("Message")
    user = relationship("User")


class ConversationResponse(BaseModel):
    id: str
    title: str
    model_type: str
    created_at: str
    updated_at: str
    message_count: int
    related_chats: Optional[List['MessageResponse']] = None

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    image_urls: Optional[List[str]] = None
    created_at: str

    class Config:
        from_attributes = True


class PaginationMetadata(BaseModel):
    page: int
    limit: int
    total_count: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedConversationsResponse(BaseModel):
    data: List[ConversationResponse]
    pagination: PaginationMetadata


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    model_type: str
    created_at: str
    updated_at: str
    messages: List[MessageResponse]

    class Config:
        from_attributes = True


class FeedbackCreate(BaseModel):
    message_id: int
    feedback_type: FeedbackType
    description: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    user_id: int
    feedback_type: str
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    nickname = Column(String, nullable=True)
    job = Column(String, nullable=True)
    chatbot_preference = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User")


class UserPreferencesUpdate(BaseModel):
    nickname: Optional[str] = None
    job: Optional[str] = None
    chatbot_preference: Optional[str] = None


class UserPreferencesResponse(BaseModel):
    id: int
    user_id: int
    nickname: Optional[str]
    job: Optional[str]
    chatbot_preference: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
