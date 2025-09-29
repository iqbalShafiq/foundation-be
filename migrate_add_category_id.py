#!/usr/bin/env python3
"""
Migration script to add category_id column to conversations table.

This script adds a nullable foreign key column category_id to the conversations table
to support user model categories feature.

Usage:
    python migrate_add_category_id.py
"""

import sqlite3
import sys
from pathlib import Path

def migrate_database(db_path: str = "app.db"):
    """Add category_id column to conversations table"""
    
    print(f"Starting migration: adding category_id column to conversations table")
    print(f"Database: {db_path}")
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"❌ Database file {db_path} not found!")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'category_id' in columns:
            print("✅ Column category_id already exists in conversations table")
            conn.close()
            return True
        
        print("📝 Adding category_id column to conversations table...")
        
        # Add the new column
        alter_sql = """
        ALTER TABLE conversations 
        ADD COLUMN category_id INTEGER 
        REFERENCES user_model_categories(id)
        """
        
        cursor.execute(alter_sql)
        conn.commit()
        
        # Verify the column was added
        cursor.execute("PRAGMA table_info(conversations)")
        columns_after = [column[1] for column in cursor.fetchall()]
        
        if 'category_id' in columns_after:
            print("✅ Successfully added category_id column to conversations table")
            
            # Show table structure
            print("\n📋 Updated conversations table structure:")
            cursor.execute("PRAGMA table_info(conversations)")
            for row in cursor.fetchall():
                print(f"  - {row[1]} ({row[2]}) {'NOT NULL' if row[3] else 'NULL'}")
            
            conn.close()
            return True
        else:
            print("❌ Failed to add category_id column")
            conn.close()
            return False
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main migration function"""
    print("🔄 Database Migration: Add category_id to conversations")
    print("=" * 60)
    
    # Check for database path argument
    db_path = sys.argv[1] if len(sys.argv) > 1 else "app.db"
    
    success = migrate_database(db_path)
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("💡 You can now use category_id in conversations")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()