from pydantic import BaseModel, EmailStr
from typing import Optional, List, Union, Dict
import uuid
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Float
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
    parent_conversation_id = Column(String, ForeignKey("conversations.id"), nullable=True, index=True)
    edited_message_id = Column(Integer, nullable=True)  # Which message was edited to create this branch
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User")
    parent_conversation = relationship("Conversation", remote_side=[id], backref="child_conversations")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    image_urls = Column(Text, nullable=True)  # JSON array of image URLs
    document_context = Column(Text, nullable=True)  # JSON object with document context info
    # Branching fields
    parent_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)
    branch_id = Column(String, nullable=True, index=True)  # UUID for branch identification
    is_active_branch = Column(Boolean, default=True)  # Whether this message is in the active branch
    # Token usage fields
    input_tokens = Column(Integer, nullable=True)  # Prompt/input tokens
    output_tokens = Column(Integer, nullable=True)  # Completion/output tokens
    total_tokens = Column(Integer, nullable=True)  # Total tokens used
    model_cost = Column(Float, nullable=True)  # Cost in USD (optional)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    conversation = relationship("Conversation")
    parent_message = relationship("Message", remote_side=[id], backref="child_messages")


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
    parent_conversation_id: Optional[str] = None
    edited_message_id: Optional[int] = None
    is_branch: Optional[bool] = False  # True if this is a branched conversation
    related_chats: Optional[List['MessageResponse']] = None

    class Config:
        from_attributes = True


class DocumentContextInfo(BaseModel):
    document_id: str
    title: str
    url: Optional[str] = None
    file_extension: Optional[str] = None

class MessageDocumentContext(BaseModel):
    collection_id: Optional[str] = None
    documents: List[DocumentContextInfo] = []
    context_chunks_count: int = 0

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    image_urls: Optional[List[str]] = None
    document_context: Optional[MessageDocumentContext] = None
    chart_data: Optional[Dict] = None  # Chart data from generate_chart tool
    # Branching fields
    parent_message_id: Optional[int] = None
    branch_id: Optional[str] = None
    is_active_branch: Optional[bool] = True
    has_branches: Optional[bool] = False  # Whether this message has alternative branches
    # Token usage fields
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_cost: Optional[float] = None
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
    parent_conversation_id: Optional[str] = None
    edited_message_id: Optional[int] = None
    is_branch: Optional[bool] = False
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


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    CSV = "csv"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    processing_status = Column(String, default=DocumentStatus.PENDING.value)
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(Text, nullable=True)  # JSON metadata
    vector_id = Column(String, nullable=True)  # ChromaDB collection ID
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    document = relationship("Document")


class DocumentCollection(Base):
    __tablename__ = "document_collections"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    document_ids = Column(Text, nullable=False)  # JSON array of doc IDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User")


class ConversationContext(Base):
    __tablename__ = "conversation_contexts"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    document_ids = Column(Text, nullable=True)  # JSON array of active contexts
    collection_id = Column(String, ForeignKey("document_collections.id"), nullable=True)
    context_settings = Column(Text, nullable=True)  # JSON: {max_chunks, threshold, etc}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    conversation = relationship("Conversation")
    collection = relationship("DocumentCollection")


# Pydantic models for API responses
class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    processing_status: str
    chunk_count: int
    error_message: Optional[str]
    document_url: Optional[str]  # URL to access the document
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class DocumentCollectionResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    document_ids: List[str]
    document_count: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class DocumentCollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    document_ids: List[str] = []


class DocumentCollectionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    document_ids: Optional[List[str]] = None


class DocumentSearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    collection_id: Optional[str] = None
    limit: int = 10
    relevance_threshold: float = 0.7


class DocumentSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    page_number: Optional[int]
    relevance_score: float
    metadata: Optional[dict]


class ContextSource(BaseModel):
    document_id: str
    document_name: str
    chunk_text: str
    page_number: Optional[int]
    relevance_score: float


class MessageEditRequest(BaseModel):
    content: str


class MessageEditResponse(BaseModel):
    message_id: int
    new_branch_id: str
    conversation_id: str
    regenerated_messages: List[MessageResponse]


class MessageBranch(BaseModel):
    branch_id: str
    root_message_id: int
    is_active: bool
    created_at: str
    message_count: int


class ConversationBranchesResponse(BaseModel):
    conversation_id: str
    branches: List[MessageBranch]
