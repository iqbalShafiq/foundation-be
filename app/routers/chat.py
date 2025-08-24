from fastapi import APIRouter, HTTPException, Depends, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, cast, List
import base64
import uuid
import json
from pathlib import Path

from app.models import (
    User,
    ConversationDetailResponse,
    PaginatedConversationsResponse,
    ImageData,
    ModelType,
    ContextSource,
)
from app.services.chat_router_service import chat_router_service
from app.services.conversation_service import ConversationService
from app.services.rag_service import RAGService
from app.dependencies import require_user_or_admin
from app.database import get_db

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(
    message: str = Form(...),
    model: ModelType = Form(ModelType.STANDARD),
    conversation_id: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None),
    # NEW: Document context parameters
    document_contexts: Optional[str] = Form(None),  # JSON string of document IDs
    context_collection: Optional[str] = Form(None),  # Collection ID
    max_context_chunks: int = Form(10),  # Maximum chunks to retrieve
    context_relevance_threshold: float = Form(0.1),  # Similarity threshold (lowered for better recall)
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Streaming chat endpoint with multipart form support for images and document context"""
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Parse document context parameters
    parsed_document_ids = None
    if document_contexts:
        try:
            parsed_document_ids = json.loads(document_contexts)
            if not isinstance(parsed_document_ids, list):
                raise ValueError("document_contexts must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid document_contexts format: {e}")

    # Retrieve document context if specified
    context_sources: Optional[List[ContextSource]] = None
    if parsed_document_ids or context_collection:
        try:
            rag_service = RAGService(db)
            context_sources = rag_service.retrieve_context(
                query=message,
                user_id=cast(int, current_user.id),
                document_ids=parsed_document_ids,
                collection_id=context_collection,
                max_chunks=max_context_chunks,
                relevance_threshold=context_relevance_threshold
            )
        except Exception as e:
            # Log error but don't fail the chat - just proceed without context
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error retrieving document context: {e}")
            logger.exception("Full traceback:")
            context_sources = None

    # Save uploaded images and create ImageData format
    image_data_list = None
    if images:
        image_data_list = []
        uploads_dir = Path("uploads/images")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        for image_file in images:
            if not image_file.content_type or not image_file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {image_file.filename} is not an image")
            
            # Generate unique filename
            file_extension = Path(image_file.filename).suffix if image_file.filename else '.jpg'
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = uploads_dir / unique_filename
            
            # Save file to disk
            image_content = await image_file.read()
            with open(file_path, "wb") as f:
                f.write(image_content)
            
            # Create URL for the saved image
            image_url = f"/uploads/images/{unique_filename}"
            
            # Still encode to base64 for OpenAI API compatibility
            base64_data = base64.b64encode(image_content).decode('utf-8')
            
            image_data_list.append(ImageData(
                data=base64_data,
                mime_type=image_file.content_type,
                url=image_url  # Add URL field
            ))

    return StreamingResponse(
        chat_router_service.route_chat_request(
            message, model, conversation_id, cast(int, current_user.id), image_data_list, context_sources, context_collection, parsed_document_ids
        ),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )



@router.get("/conversations", response_model=PaginatedConversationsResponse)
async def get_user_conversations(
    keyword: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db),
):
    """Get all conversations for the current user, optionally filtered by keyword search with pagination"""
    conversation_service = ConversationService(db)
    return conversation_service.get_user_conversations(
        user_id=cast(int, current_user.id), 
        keyword=keyword,
        page=page,
        limit=limit
    )


@router.get(
    "/conversations/{conversation_id}", response_model=ConversationDetailResponse
)
async def get_conversation_detail(
    conversation_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db),
):
    """Get detailed conversation with full message history"""
    conversation_service = ConversationService(db)
    result = conversation_service.get_conversation_detail(
        conversation_id=conversation_id,
        user_id=cast(int, current_user.id)
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation, messages = result
    
    return ConversationDetailResponse.model_validate(
        {
            "id": conversation.id,
            "title": conversation.title,
            "model_type": conversation.model_type,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": [
                conversation_service._message_to_response(msg)
                for msg in messages
            ],
        }
    )
