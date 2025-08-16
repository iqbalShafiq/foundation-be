from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, cast

from app.models import (
    ChatRequest,
    User,
    ConversationDetailResponse,
    MessageResponse,
    PaginatedConversationsResponse,
)
from app.services.chat_service import chat_service
from app.services.conversation_service import ConversationService
from app.dependencies import require_user_or_admin
from app.database import get_db

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest, current_user: User = Depends(require_user_or_admin)
):
    """Streaming chat endpoint"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return StreamingResponse(
        chat_service.generate_stream_response(
            request.message, request.model, str(request.conversation_id), cast(int, current_user.id)
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
                MessageResponse.model_validate(
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat(),
                    }
                )
                for msg in messages
            ],
        }
    )
