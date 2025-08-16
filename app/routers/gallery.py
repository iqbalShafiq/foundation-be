from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, cast, List
import json

from app.models import (
    User,
    Message,
    PaginationMetadata
)
from app.dependencies import require_user_or_admin, require_admin
from app.database import get_db
from pydantic import BaseModel


class ImageItem(BaseModel):
    url: str
    message_id: int
    conversation_id: str
    created_at: str
    message_content: str


class GalleryResponse(BaseModel):
    data: List[ImageItem]
    pagination: PaginationMetadata


router = APIRouter(tags=["gallery"])


@router.get("/gallery/user/{user_id}", response_model=GalleryResponse)
async def get_user_gallery(
    user_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db),
):
    """Get all uploaded images by a specific user with pagination (admin only or own user)"""
    
    # Check if current user can access this gallery
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=403, 
            detail="You can only access your own gallery"
        )
    
    # Check if target user exists
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate offset
    offset = (page - 1) * limit
    
    # Query messages with images from this user
    query = db.query(Message).join(
        Message.conversation
    ).filter(
        Message.conversation.has(user_id=user_id),
        Message.image_urls.isnot(None),
        Message.image_urls != "null",
        Message.image_urls != ""
    ).order_by(Message.created_at.desc())
    
    # Get total count
    total_count = query.count()
    
    # Get paginated results
    messages = query.offset(offset).limit(limit).all()
    
    # Extract image URLs and create response
    image_items = []
    for msg in messages:
        if msg.image_urls:
            try:
                urls = json.loads(msg.image_urls)
                for url in urls:
                    image_items.append(ImageItem(
                        url=url,
                        message_id=msg.id,
                        conversation_id=msg.conversation_id,
                        created_at=msg.created_at.isoformat(),
                        message_content=msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    ))
            except json.JSONDecodeError:
                continue
    
    # Calculate pagination metadata
    total_pages = (total_count + limit - 1) // limit
    
    pagination = PaginationMetadata(
        page=page,
        limit=limit,
        total_count=total_count,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )
    
    return GalleryResponse(
        data=image_items,
        pagination=pagination
    )


@router.get("/gallery/me", response_model=GalleryResponse)
async def get_my_gallery(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db),
):
    """Get all uploaded images by current user with pagination"""
    
    # Calculate offset
    offset = (page - 1) * limit
    
    # Query messages with images from current user
    query = db.query(Message).join(
        Message.conversation
    ).filter(
        Message.conversation.has(user_id=current_user.id),
        Message.image_urls.isnot(None),
        Message.image_urls != "null", 
        Message.image_urls != ""
    ).order_by(Message.created_at.desc())
    
    # Get total count
    total_count = query.count()
    
    # Get paginated results
    messages = query.offset(offset).limit(limit).all()
    
    # Extract image URLs and create response
    image_items = []
    for msg in messages:
        if msg.image_urls:
            try:
                urls = json.loads(msg.image_urls)
                for url in urls:
                    image_items.append(ImageItem(
                        url=url,
                        message_id=msg.id,
                        conversation_id=msg.conversation_id,
                        created_at=msg.created_at.isoformat(),
                        message_content=msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    ))
            except json.JSONDecodeError:
                continue
    
    # Calculate pagination metadata
    total_pages = (total_count + limit - 1) // limit
    
    pagination = PaginationMetadata(
        page=page,
        limit=limit,
        total_count=total_count,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )
    
    return GalleryResponse(
        data=image_items,
        pagination=pagination
    )


@router.get("/gallery/all", response_model=GalleryResponse)
async def get_all_gallery(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get all uploaded images from all users with pagination (admin only)"""
    
    # Calculate offset
    offset = (page - 1) * limit
    
    # Query all messages with images
    query = db.query(Message).filter(
        Message.image_urls.isnot(None),
        Message.image_urls != "null",
        Message.image_urls != ""
    ).order_by(Message.created_at.desc())
    
    # Get total count
    total_count = query.count()
    
    # Get paginated results
    messages = query.offset(offset).limit(limit).all()
    
    # Extract image URLs and create response
    image_items = []
    for msg in messages:
        if msg.image_urls:
            try:
                urls = json.loads(msg.image_urls)
                for url in urls:
                    image_items.append(ImageItem(
                        url=url,
                        message_id=msg.id,
                        conversation_id=msg.conversation_id,
                        created_at=msg.created_at.isoformat(),
                        message_content=msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    ))
            except json.JSONDecodeError:
                continue
    
    # Calculate pagination metadata
    total_pages = (total_count + limit - 1) // limit
    
    pagination = PaginationMetadata(
        page=page,
        limit=limit,
        total_count=total_count,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )
    
    return GalleryResponse(
        data=image_items,
        pagination=pagination
    )