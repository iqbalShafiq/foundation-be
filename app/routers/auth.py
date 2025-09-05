from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import (
    UserCreate, UserLogin, Token, UserResponse, User, UserUpdate,
    MonthlyDailyBreakdownResponse, DailyConversationBreakdownResponse
)
from ..services.auth_service import AuthService, ACCESS_TOKEN_EXPIRE_MINUTES
from ..services.token_aggregation_service import TokenAggregationService
from ..dependencies import get_current_active_user, require_admin
from typing import List, Optional

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    return AuthService.create_user(db=db, user=user)


@router.post("/login", response_model=Token)
def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    user = AuthService.authenticate_user(
        db, user_credentials.username, user_credentials.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = AuthService.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def read_users_me(
    current_user: User = Depends(get_current_active_user),
    limit: int = Query(12, ge=1, le=36, description="Number of months to return (max 36)"),
    from_year: Optional[int] = Query(None, description="Starting year for pagination"),
    from_month: Optional[int] = Query(None, ge=1, le=12, description="Starting month for pagination"),
    include_token_stats: bool = Query(True, description="Include token statistics"),
    db: Session = Depends(get_db)
):
    # Validate from_month if from_year is provided
    if from_year and not from_month:
        raise HTTPException(status_code=400, detail="from_month is required when from_year is provided")
    
    # Get token statistics if requested
    token_stats = None
    if include_token_stats:
        token_stats = AuthService.get_user_token_stats(
            db=db,
            user_id=current_user.id,
            limit=limit,
            from_year=from_year,
            from_month=from_month
        )
    
    # Create response
    user_response = UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        token_stats=token_stats
    )
    
    return user_response


@router.get("/users", response_model=List[UserResponse])
def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@router.put("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    return AuthService.update_user(
        db=db, user_id=user_id, user_update=user_update
    )


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}


@router.get("/token-stats/monthly/{year}/{month}/daily", response_model=MonthlyDailyBreakdownResponse)
def get_monthly_daily_breakdown(
    year: int,
    month: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get daily breakdown of token usage within a specific month"""
    
    # Validate month
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    # Validate year (reasonable range)
    current_year = datetime.now().year
    if not (2020 <= year <= current_year + 1):
        raise HTTPException(status_code=400, detail=f"Year must be between 2020 and {current_year + 1}")
    
    return TokenAggregationService.get_monthly_daily_breakdown(
        db=db, 
        user_id=current_user.id, 
        year=year, 
        month=month
    )


@router.get("/token-stats/daily/{date}/conversations", response_model=DailyConversationBreakdownResponse)
def get_daily_conversation_breakdown(
    date: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get conversation-level breakdown for a specific day (format: YYYY-MM-DD)"""
    
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")
    
    # Validate date is not in the future
    if parsed_date.date() > datetime.now().date():
        raise HTTPException(status_code=400, detail="Date cannot be in the future")
    
    return TokenAggregationService.get_daily_conversation_breakdown(
        db=db,
        user_id=current_user.id,
        date=parsed_date
    )
