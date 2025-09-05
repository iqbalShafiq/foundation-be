from datetime import datetime, timedelta
from typing import Optional, cast
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from fastapi import HTTPException, status
from ..models import User, UserCreate, UserUpdate, TokenData, Message, MonthlyTokenStats, UserTokenStatsResponse
from .token_aggregation_service import TokenAggregationService
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now() + expires_delta
        else:
            expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> TokenData:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = cast(str, payload.get("sub"))
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return TokenData(username=username)
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def create_user(db: Session, user: UserCreate) -> User:
        existing_user = AuthService.get_user_by_username(db, user.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
            )

        existing_email = AuthService.get_user_by_email(db, user.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        hashed_password = AuthService.get_password_hash(user.password)
        db_user = User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            role=user.role.value,
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        user = AuthService.get_user_by_username(db, username)
        if not user:
            return None
        if not AuthService.verify_password(password, cast(str, user.hashed_password)):
            return None
        return user

    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def update_user(db: Session, user_id: int, user_update: UserUpdate) -> User:
        user = AuthService.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        if user_update.role is not None:
            setattr(user, "role", user_update.role.value)

        if user_update.password is not None:
            setattr(
                user,
                "hashed_password",
                AuthService.get_password_hash(user_update.password),
            )

        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_monthly_token_stats(db: Session, user_id: int, year: int, month: int) -> MonthlyTokenStats:
        """Get aggregated token statistics for a user for a specific month"""
        # Query messages where user owns the conversation and messages have token data
        result = (
            db.query(
                func.sum(Message.input_tokens).label("total_input_tokens"),
                func.sum(Message.output_tokens).label("total_output_tokens"), 
                func.sum(Message.total_tokens).label("total_total_tokens"),
                func.sum(Message.model_cost).label("total_cost"),
                func.count(Message.id).label("message_count"),
            )
            .join(Message.conversation)  # Join with Conversation table
            .filter(
                Message.role == "assistant",  # Only count AI assistant messages for token usage
                Message.input_tokens.isnot(None),  # Has token data
                Message.conversation.has(user_id=user_id),  # User owns the conversation
                extract('year', Message.created_at) == year,
                extract('month', Message.created_at) == month,
            )
            .first()
        )
        
        # Handle case where no data exists
        input_tokens = result.total_input_tokens or 0
        output_tokens = result.total_output_tokens or 0
        total_tokens = result.total_total_tokens or 0
        total_cost = float(result.total_cost or 0.0)
        message_count = result.message_count or 0
        
        return MonthlyTokenStats(
            month=f"{year:04d}-{month:02d}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            total_cost=total_cost,
            message_count=message_count,
        )

    @staticmethod
    def get_user_token_stats(
        db: Session, 
        user_id: int, 
        limit: int = 12,
        from_year: Optional[int] = None,
        from_month: Optional[int] = None
    ) -> UserTokenStatsResponse:
        """Get paginated monthly token statistics for a user (FAST VERSION)"""
        
        # Ensure current month is up-to-date for real-time accuracy
        TokenAggregationService.ensure_current_month_updated(db, user_id)
        
        # Use the fast pre-computed method
        return TokenAggregationService.get_user_token_stats_fast(
            db=db,
            user_id=user_id,
            limit=limit,
            from_year=from_year,
            from_month=from_month
        )
