from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import cast

from app.models import (
    User,
    MessageEditRequest,
    MessageEditResponse,
    ConversationBranchesResponse,
)
from app.services.message_service import MessageService
from app.dependencies import require_user_or_admin
from app.database import get_db

router = APIRouter(tags=["messages"])


@router.put("/messages/{message_id}", response_model=MessageEditResponse)
async def edit_message(
    message_id: int,
    edit_request: MessageEditRequest,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db),
):
    """Edit a user message and create a new branch with regenerated AI responses"""
    message_service = MessageService(db)
    
    try:
        result = message_service.edit_message(
            message_id=message_id,
            user_id=cast(int, current_user.id),
            edit_request=edit_request
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Message not found or access denied")
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to edit message: {str(e)}")


@router.get("/conversations/{conversation_id}/branches", response_model=ConversationBranchesResponse)
async def get_conversation_branches(
    conversation_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db),
):
    """Get all branches for a conversation"""
    message_service = MessageService(db)
    
    result = message_service.get_conversation_branches(
        conversation_id=conversation_id,
        user_id=cast(int, current_user.id)
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    
    return result


@router.post("/conversations/{conversation_id}/branches/{branch_id}/activate")
async def activate_branch(
    conversation_id: str,
    branch_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db),
):
    """Switch the active branch for a conversation"""
    message_service = MessageService(db)
    
    success = message_service.switch_active_branch(
        conversation_id=conversation_id,
        branch_id=branch_id,
        user_id=cast(int, current_user.id)
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    
    return {"message": f"Branch {branch_id} activated successfully"}