#!/usr/bin/env python3
"""
Migration script to populate UserMonthlyTokenUsage table with existing data.

This script should be run after the new table is created to populate historical data
for existing users who already have conversations with token usage.

Usage: python migrate_token_data.py
"""

import sys
import os
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import extract

# Add app directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, engine
from app.models import Base, Message, User, UserMonthlyTokenUsage
from app.services.token_aggregation_service import TokenAggregationService


def create_tables():
    """Ensure all tables are created"""
    print("📋 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully")


def get_users_with_token_data(db: Session):
    """Get all users who have messages with token usage data"""
    users_with_tokens = (
        db.query(User.id, User.username)
        .join(Message.conversation)
        .filter(
            Message.role == "assistant",
            Message.input_tokens.isnot(None),
            Message.conversation.has(user_id=User.id)
        )
        .distinct()
        .all()
    )
    return users_with_tokens


def get_user_month_combinations(db: Session, user_id: int):
    """Get all year-month combinations where user has token usage data"""
    month_combinations = (
        db.query(
            extract('year', Message.created_at).label('year'),
            extract('month', Message.created_at).label('month')
        )
        .join(Message.conversation)
        .filter(
            Message.role == "assistant",
            Message.input_tokens.isnot(None),
            Message.conversation.has(user_id=user_id)
        )
        .distinct()
        .order_by('year', 'month')
        .all()
    )
    return [(int(year), int(month)) for year, month in month_combinations]


def migrate_user_token_data(db: Session, user_id: int, username: str):
    """Migrate token data for a specific user"""
    print(f"🔄 Processing user: {username} (ID: {user_id})")
    
    # Get all months where this user has token data
    month_combinations = get_user_month_combinations(db, user_id)
    
    if not month_combinations:
        print(f"   ⚠️  No token data found for {username}")
        return 0
    
    migrated_count = 0
    
    for year, month in month_combinations:
        try:
            # Check if this month already exists
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
                print(f"   ⏭️  {year}-{month:02d} already exists, skipping")
                continue
            
            # Create/update the monthly summary
            summary = TokenAggregationService.update_monthly_token_usage(
                db, user_id, year, month
            )
            
            if summary and summary.message_count > 0:
                print(f"   ✅ {year}-{month:02d}: {summary.message_count} messages, "
                      f"{summary.total_tokens} tokens, ${summary.total_cost:.4f}")
                migrated_count += 1
            else:
                print(f"   ⚠️  {year}-{month:02d}: No valid data")
                
        except Exception as e:
            print(f"   ❌ Error processing {year}-{month:02d}: {e}")
    
    return migrated_count


def main():
    """Main migration function"""
    print("🚀 Starting Token Usage Data Migration")
    print("=" * 50)
    
    # Create tables first
    create_tables()
    
    db = SessionLocal()
    try:
        # Get all users with token data
        users_with_tokens = get_users_with_token_data(db)
        
        if not users_with_tokens:
            print("📋 No users with token data found. Migration complete!")
            return
        
        print(f"📊 Found {len(users_with_tokens)} users with token usage data")
        print()
        
        total_migrated = 0
        
        for user_id, username in users_with_tokens:
            migrated_count = migrate_user_token_data(db, user_id, username)
            total_migrated += migrated_count
            print()
        
        print("=" * 50)
        print(f"🎉 Migration completed successfully!")
        print(f"📈 Migrated {total_migrated} monthly summaries across {len(users_with_tokens)} users")
        
        # Show some statistics
        total_summaries = db.query(UserMonthlyTokenUsage).count()
        total_tokens = db.query(
            db.func.sum(UserMonthlyTokenUsage.total_tokens)
        ).scalar() or 0
        total_cost = db.query(
            db.func.sum(UserMonthlyTokenUsage.total_cost)
        ).scalar() or 0.0
        
        print(f"📊 Database now contains:")
        print(f"   • {total_summaries} monthly summaries")
        print(f"   • {total_tokens:,} total tokens processed")
        print(f"   • ${total_cost:.4f} total cost tracked")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.rollback()
        sys.exit(1)
    
    finally:
        db.close()


if __name__ == "__main__":
    main()