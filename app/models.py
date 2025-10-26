from pydantic import BaseModel, EmailStr
from typing import Optional, List, Union, Dict
import uuid
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Float, UniqueConstraint
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
    model_id: str = "anthropic/claude-sonnet-4"  # Accept model ID directly
    category_id: Optional[int] = None  # Optional category ID for UI display
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


class MonthlyTokenStats(BaseModel):
    month: str  # Format: YYYY-MM
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    message_count: int


class UserTokenStatsResponse(BaseModel):
    monthly_stats: List[MonthlyTokenStats]
    total_months: int
    has_more: bool


class DailyTokenStats(BaseModel):
    date: str  # Format: YYYY-MM-DD
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    message_count: int
    conversation_count: int


class MonthlyDailyBreakdownResponse(BaseModel):
    month: str  # Format: YYYY-MM
    daily_stats: List[DailyTokenStats]
    total_days: int


class ConversationTokenStats(BaseModel):
    conversation_id: str
    conversation_title: str
    model_type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    message_count: int
    last_message_at: str


class DailyConversationBreakdownResponse(BaseModel):
    date: str  # Format: YYYY-MM-DD
    conversation_stats: List[ConversationTokenStats]
    total_conversations: int


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    token_stats: Optional[UserTokenStatsResponse] = None

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
    category_id = Column(Integer, ForeignKey("user_model_categories.id"), nullable=True)
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
    category_id: Optional[int] = None
    category_name: Optional[str] = None
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
    chart_data: Optional[Union[Dict, List[Dict]]] = None  # Support both single chart (backward compat) and multiple charts
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
    category_id: Optional[int] = None
    category_name: Optional[str] = None
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


# Pre-computed monthly token usage table for performance
class UserMonthlyTokenUsage(Base):
    __tablename__ = "user_monthly_token_usage"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False, index=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    message_count = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User")
    
    # Ensure unique constraint on user_id + year + month
    __table_args__ = (
        UniqueConstraint('user_id', 'year', 'month', name='unique_user_month'),
    )


# Model metadata from OpenRouter API
class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(String, primary_key=True, index=True)  # model ID from OpenRouter
    canonical_slug = Column(String, nullable=True, index=True)
    hugging_face_id = Column(String, nullable=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    context_length = Column(Integer, nullable=True)
    
    # Architecture info
    modality = Column(String, nullable=True)  # e.g., "text->text", "text+image->text"
    input_modalities = Column(Text, nullable=True)  # JSON array
    output_modalities = Column(Text, nullable=True)  # JSON array
    tokenizer = Column(String, nullable=True)
    instruct_type = Column(String, nullable=True)
    
    # Pricing info
    prompt_price = Column(Float, nullable=True)  # Price per token for prompt
    completion_price = Column(Float, nullable=True)  # Price per token for completion
    request_price = Column(Float, nullable=True)  # Price per request
    image_price = Column(Float, nullable=True)  # Price per image
    web_search_price = Column(Float, nullable=True)  # Price for web search
    internal_reasoning_price = Column(Float, nullable=True)  # Price for internal reasoning
    input_cache_read_price = Column(Float, nullable=True)  # Price for input cache read
    
    # Provider info
    provider_context_length = Column(Integer, nullable=True)
    max_completion_tokens = Column(Integer, nullable=True)
    is_moderated = Column(Boolean, default=False)
    
    # Additional metadata
    supported_parameters = Column(Text, nullable=True)  # JSON array
    per_request_limits = Column(Text, nullable=True)  # JSON object
    
    # Internal tracking
    is_active = Column(Boolean, default=True)  # Whether model is currently available
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Pydantic models for API responses
class ModelMetadataResponse(BaseModel):
    id: str
    canonical_slug: Optional[str]
    hugging_face_id: Optional[str]
    name: str
    description: Optional[str]
    context_length: Optional[int]
    modality: Optional[str]
    input_modalities: Optional[List[str]]
    output_modalities: Optional[List[str]]
    tokenizer: Optional[str]
    instruct_type: Optional[str]
    prompt_price: Optional[float]
    completion_price: Optional[float]
    request_price: Optional[float]
    image_price: Optional[float]
    web_search_price: Optional[float]
    internal_reasoning_price: Optional[float]
    input_cache_read_price: Optional[float]
    provider_context_length: Optional[int]
    max_completion_tokens: Optional[int]
    is_moderated: Optional[bool]
    supported_parameters: Optional[List[str]]
    per_request_limits: Optional[Dict]
    is_active: bool
    last_updated: str
    created_at: str

    class Config:
        from_attributes = True


class ModelMetadataListResponse(BaseModel):
    models: List[ModelMetadataResponse]
    total_count: int
    updated_at: str


# User Model Categories - Customizable model preferences per user
class UserModelCategory(Base):
    __tablename__ = "user_model_categories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    category_name = Column(String, nullable=False)  # e.g., "Fast", "Standard", "Heavy Work"
    display_name = Column(String, nullable=False)  # Display name for UI
    model_id = Column(String, ForeignKey("model_metadata.id"), nullable=False)  # Reference to ModelMetadata
    description = Column(Text, nullable=True)  # User description of when to use this
    sort_order = Column(Integer, default=0)  # For ordering in UI
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User")
    model_metadata = relationship("ModelMetadata")
    
    # Ensure unique category name per user
    __table_args__ = (
        UniqueConstraint('user_id', 'category_name', name='unique_user_category'),
    )


# Pydantic models for User Model Categories
class UserModelCategoryCreate(BaseModel):
    category_name: str
    display_name: str
    model_id: str
    description: Optional[str] = None
    sort_order: Optional[int] = 0


class UserModelCategoryUpdate(BaseModel):
    category_name: Optional[str] = None
    display_name: Optional[str] = None
    model_id: Optional[str] = None
    description: Optional[str] = None
    sort_order: Optional[int] = None
    is_active: Optional[bool] = None


class UserModelCategoryResponse(BaseModel):
    id: int
    category_name: str
    display_name: str
    model_id: str
    model_name: Optional[str] = None  # From ModelMetadata
    model_pricing: Optional[Dict] = None  # Basic pricing info
    description: Optional[str]
    sort_order: int
    is_active: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class UserModelCategoriesListResponse(BaseModel):
    categories: List[UserModelCategoryResponse]
    total_count: int
