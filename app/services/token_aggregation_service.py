from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, extract, or_
from ..models import (
    Message, UserMonthlyTokenUsage, MonthlyTokenStats, UserTokenStatsResponse,
    DailyTokenStats, MonthlyDailyBreakdownResponse, ConversationTokenStats, 
    DailyConversationBreakdownResponse, Conversation
)
import logging

logger = logging.getLogger(__name__)


class TokenAggregationService:
    """Service for efficiently managing token usage aggregation"""
    
    @staticmethod
    def update_monthly_token_usage(db: Session, user_id: int, year: int, month: int) -> UserMonthlyTokenUsage:
        """Update or create monthly token usage summary for a specific user/month"""
        
        # Get aggregated data from messages for this month
        result = (
            db.query(
                func.sum(Message.input_tokens).label("total_input_tokens"),
                func.sum(Message.output_tokens).label("total_output_tokens"), 
                func.sum(Message.total_tokens).label("total_total_tokens"),
                func.sum(Message.model_cost).label("total_cost"),
                func.count(Message.id).label("message_count"),
            )
            .join(Message.conversation)
            .filter(
                Message.role == "assistant",
                Message.input_tokens.isnot(None),
                Message.conversation.has(user_id=user_id),
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
        
        # Find existing record or create new one
        existing = (
            db.query(UserMonthlyTokenUsage)
            .filter(
                UserMonthlyTokenUsage.user_id == user_id,
                UserMonthlyTokenUsage.year == year,
                UserMonthlyTokenUsage.month == month
            )
            .first()
        )
        
        if existing:
            # Update existing record
            existing.input_tokens = input_tokens
            existing.output_tokens = output_tokens
            existing.total_tokens = total_tokens
            existing.total_cost = total_cost
            existing.message_count = message_count
            existing.last_updated = func.now()
            
            db.commit()
            return existing
        else:
            # Create new record (only if there's actual usage)
            if message_count > 0:
                new_usage = UserMonthlyTokenUsage(
                    user_id=user_id,
                    year=year,
                    month=month,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    message_count=message_count
                )
                db.add(new_usage)
                db.commit()
                db.refresh(new_usage)
                return new_usage
            
            # Return empty record for consistency
            return UserMonthlyTokenUsage(
                user_id=user_id, year=year, month=month,
                input_tokens=0, output_tokens=0, total_tokens=0, 
                total_cost=0.0, message_count=0
            )
    
    @staticmethod
    def update_user_token_stats_for_new_message(db: Session, user_id: int, message_created_at: datetime):
        """Update monthly stats when a new message with tokens is created"""
        year = message_created_at.year
        month = message_created_at.month
        
        try:
            TokenAggregationService.update_monthly_token_usage(db, user_id, year, month)
            logger.info(f"Updated token stats for user {user_id}, {year}-{month:02d}")
        except Exception as e:
            logger.error(f"Failed to update token stats for user {user_id}: {e}")
    
    @staticmethod
    def get_user_token_stats_fast(
        db: Session, 
        user_id: int, 
        limit: int = 12,
        from_year: Optional[int] = None,
        from_month: Optional[int] = None
    ) -> UserTokenStatsResponse:
        """Get token statistics using pre-computed summaries (FAST!)"""
        
        # Build query for pre-computed data
        query = db.query(UserMonthlyTokenUsage).filter(
            UserMonthlyTokenUsage.user_id == user_id,
            UserMonthlyTokenUsage.message_count > 0  # Only months with actual usage
        )
        
        # Apply from_year/from_month filter if provided
        if from_year and from_month:
            query = query.filter(
                or_(
                    UserMonthlyTokenUsage.year > from_year,
                    (UserMonthlyTokenUsage.year == from_year) & (UserMonthlyTokenUsage.month >= from_month)
                )
            )
        
        # Order by year, month DESC (most recent first) and apply limit
        monthly_records = (
            query
            .order_by(UserMonthlyTokenUsage.year.desc(), UserMonthlyTokenUsage.month.desc())
            .limit(limit + 1)  # Get one extra to check if there are more
            .all()
        )
        
        # Check if there are more months beyond the limit
        has_more = len(monthly_records) > limit
        if has_more:
            monthly_records = monthly_records[:limit]
        
        # Convert to MonthlyTokenStats format
        monthly_stats = []
        for record in monthly_records:
            stats = MonthlyTokenStats(
                month=f"{record.year:04d}-{record.month:02d}",
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                total_tokens=record.total_tokens,
                total_cost=record.total_cost,
                message_count=record.message_count
            )
            monthly_stats.append(stats)
        
        # Get total count of months with data
        total_months_query = db.query(UserMonthlyTokenUsage).filter(
            UserMonthlyTokenUsage.user_id == user_id,
            UserMonthlyTokenUsage.message_count > 0
        )
        
        if from_year and from_month:
            total_months_query = total_months_query.filter(
                or_(
                    UserMonthlyTokenUsage.year > from_year,
                    (UserMonthlyTokenUsage.year == from_year) & (UserMonthlyTokenUsage.month >= from_month)
                )
            )
        
        total_months = total_months_query.count()
        
        return UserTokenStatsResponse(
            monthly_stats=monthly_stats,
            total_months=total_months,
            has_more=has_more
        )
    
    @staticmethod
    def ensure_current_month_updated(db: Session, user_id: int):
        """Ensure current month's stats are up-to-date (for real-time accuracy)"""
        now = datetime.now()
        TokenAggregationService.update_monthly_token_usage(db, user_id, now.year, now.month)
    
    @staticmethod
    def bulk_update_all_users_current_month(db: Session):
        """Background task: Update current month stats for all users with recent activity"""
        now = datetime.now()
        
        # Find users with messages in current month
        users_with_current_month_activity = (
            db.query(Message.conversation.has(user_id=Message.user_id))
            .join(Message.conversation)
            .filter(
                Message.role == "assistant",
                Message.input_tokens.isnot(None),
                extract('year', Message.created_at) == now.year,
                extract('month', Message.created_at) == now.month,
            )
            .distinct()
            .all()
        )
        
        logger.info(f"Updating current month token stats for {len(users_with_current_month_activity)} users")
        
        for user_id in users_with_current_month_activity:
            try:
                TokenAggregationService.update_monthly_token_usage(db, user_id, now.year, now.month)
            except Exception as e:
                logger.error(f"Failed to update stats for user {user_id}: {e}")
    
    @staticmethod
    def get_monthly_daily_breakdown(
        db: Session, 
        user_id: int, 
        year: int, 
        month: int
    ) -> MonthlyDailyBreakdownResponse:
        """Get daily breakdown of token usage within a specific month"""
        
        # Query for daily aggregated data
        daily_results = (
            db.query(
                func.date(Message.created_at).label("message_date"),
                func.sum(Message.input_tokens).label("total_input_tokens"),
                func.sum(Message.output_tokens).label("total_output_tokens"),
                func.sum(Message.total_tokens).label("total_total_tokens"),
                func.sum(Message.model_cost).label("total_cost"),
                func.count(Message.id).label("message_count"),
                func.count(func.distinct(Message.conversation_id)).label("conversation_count")
            )
            .join(Message.conversation)
            .filter(
                Message.role == "assistant",
                Message.input_tokens.isnot(None),
                Message.conversation.has(user_id=user_id),
                extract('year', Message.created_at) == year,
                extract('month', Message.created_at) == month
            )
            .group_by(func.date(Message.created_at))
            .order_by(func.date(Message.created_at).desc())
            .all()
        )
        
        # Convert to DailyTokenStats format
        daily_stats = []
        for result in daily_results:
            stats = DailyTokenStats(
                date=str(result.message_date),
                input_tokens=result.total_input_tokens or 0,
                output_tokens=result.total_output_tokens or 0,
                total_tokens=result.total_total_tokens or 0,
                total_cost=float(result.total_cost or 0.0),
                message_count=result.message_count or 0,
                conversation_count=result.conversation_count or 0
            )
            daily_stats.append(stats)
        
        return MonthlyDailyBreakdownResponse(
            month=f"{year:04d}-{month:02d}",
            daily_stats=daily_stats,
            total_days=len(daily_stats)
        )
    
    @staticmethod
    def get_daily_conversation_breakdown(
        db: Session,
        user_id: int,
        date: datetime
    ) -> DailyConversationBreakdownResponse:
        """Get conversation-level breakdown for a specific day"""
        
        # Start and end of the day
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        # Query for conversation-level aggregated data
        conversation_results = (
            db.query(
                Message.conversation_id,
                Conversation.title.label("conversation_title"),
                Conversation.model_type,
                func.sum(Message.input_tokens).label("total_input_tokens"),
                func.sum(Message.output_tokens).label("total_output_tokens"),
                func.sum(Message.total_tokens).label("total_total_tokens"),
                func.sum(Message.model_cost).label("total_cost"),
                func.count(Message.id).label("message_count"),
                func.max(Message.created_at).label("last_message_at")
            )
            .join(Message.conversation)
            .filter(
                Message.role == "assistant",
                Message.input_tokens.isnot(None),
                Message.conversation.has(user_id=user_id),
                Message.created_at >= start_date,
                Message.created_at < end_date
            )
            .group_by(
                Message.conversation_id, 
                Conversation.title, 
                Conversation.model_type
            )
            .order_by(func.max(Message.created_at).desc())
            .all()
        )
        
        # Convert to ConversationTokenStats format
        conversation_stats = []
        for result in conversation_results:
            stats = ConversationTokenStats(
                conversation_id=result.conversation_id,
                conversation_title=result.conversation_title,
                model_type=result.model_type,
                input_tokens=result.total_input_tokens or 0,
                output_tokens=result.total_output_tokens or 0,
                total_tokens=result.total_total_tokens or 0,
                total_cost=float(result.total_cost or 0.0),
                message_count=result.message_count or 0,
                last_message_at=result.last_message_at.isoformat()
            )
            conversation_stats.append(stats)
        
        return DailyConversationBreakdownResponse(
            date=date.strftime("%Y-%m-%d"),
            conversation_stats=conversation_stats,
            total_conversations=len(conversation_stats)
        )