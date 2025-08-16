from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import (
    UserPreferences, UserPreferencesUpdate, UserPreferencesResponse, User
)
from ..dependencies import get_current_active_user

router = APIRouter(prefix="/preferences", tags=["preferences"])


@router.get("/", response_model=UserPreferencesResponse)
def get_user_preferences(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's preferences"""
    preferences = db.query(UserPreferences).filter(
        UserPreferences.user_id == current_user.id
    ).first()
    
    if not preferences:
        # Create default preferences if none exist
        preferences = UserPreferences(
            user_id=current_user.id,
            nickname=None,
            job=None,
            chatbot_preference=None
        )
        db.add(preferences)
        db.commit()
        db.refresh(preferences)
    
    return preferences


@router.put("/", response_model=UserPreferencesResponse)
def update_user_preferences(
    preferences_update: UserPreferencesUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user's preferences"""
    preferences = db.query(UserPreferences).filter(
        UserPreferences.user_id == current_user.id
    ).first()
    
    if not preferences:
        # Create new preferences if none exist
        preferences = UserPreferences(
            user_id=current_user.id,
            nickname=preferences_update.nickname,
            job=preferences_update.job,
            chatbot_preference=preferences_update.chatbot_preference
        )
        db.add(preferences)
    else:
        # Update existing preferences
        update_data = preferences_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(preferences, field, value)
    
    db.commit()
    db.refresh(preferences)
    return preferences