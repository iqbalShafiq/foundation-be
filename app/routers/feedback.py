from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..database import get_db
from ..models import Feedback, FeedbackCreate, FeedbackResponse, User, Message
from ..dependencies import get_current_user

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/", response_model=FeedbackResponse)
def create_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check if message exists
    message = db.query(Message).filter(Message.id == feedback.message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Check if user already provided feedback for this message
    existing_feedback = db.query(Feedback).filter(
        Feedback.message_id == feedback.message_id,
        Feedback.user_id == current_user.id
    ).first()
    
    if existing_feedback:
        # Update existing feedback
        existing_feedback.feedback_type = feedback.feedback_type.value
        existing_feedback.description = feedback.description
        db.commit()
        db.refresh(existing_feedback)
        return existing_feedback
    
    # Create new feedback
    db_feedback = Feedback(
        message_id=feedback.message_id,
        user_id=current_user.id,
        feedback_type=feedback.feedback_type.value,
        description=feedback.description
    )
    
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    
    return db_feedback


@router.get("/message/{message_id}", response_model=List[FeedbackResponse])
def get_message_feedback(
    message_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check if message exists
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    feedback_list = db.query(Feedback).filter(Feedback.message_id == message_id).all()
    return feedback_list


@router.delete("/{feedback_id}")
def delete_feedback(
    feedback_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Only the user who created the feedback can delete it
    if feedback.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this feedback")
    
    db.delete(feedback)
    db.commit()
    
    return {"message": "Feedback deleted successfully"}