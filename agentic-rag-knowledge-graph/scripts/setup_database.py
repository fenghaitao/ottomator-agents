#!/usr/bin/env python3
"""
Database setup script to execute schema.sql and verify the setup.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncpg

# Load environment variables
load_dotenv()

async def setup_database():
    """Setup the database by executing schema.sql"""
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        print("Please create a .env file or set DATABASE_URL environment variable")
        print("Example: DATABASE_URL=postgresql://user:password@localhost:5432/agentic_rag")
        return False
    
    print(f"🔗 Connecting to database...")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(database_url)
        print("✅ Database connection successful")
        
        # Read schema file
        schema_path = Path("sql/schema.sql")
        if not schema_path.exists():
            print(f"❌ Schema file not found: {schema_path}")
            return False
        
        print("📖 Reading schema.sql...")
        schema_sql = schema_path.read_text()
        
        # Execute schema
        print("🔧 Executing schema.sql...")
        await conn.execute(schema_sql)
        print("✅ Schema executed successfully")
        
        # Verify tables were created
        print("🔍 Verifying database setup...")
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        table_names = [row['table_name'] for row in tables]
        expected_tables = ['documents', 'chunks', 'sessions', 'messages']
        
        print(f"📋 Found tables: {', '.join(table_names)}")
        
        missing_tables = set(expected_tables) - set(table_names)
        if missing_tables:
            print(f"❌ Missing expected tables: {', '.join(missing_tables)}")
            return False
        
        # Verify extensions
        print("🧩 Verifying extensions...")
        extensions = await conn.fetch("SELECT extname FROM pg_extension")
        ext_names = [row['extname'] for row in extensions]
        
        required_extensions = ['vector', 'uuid-ossp', 'pg_trgm']
        missing_extensions = set(required_extensions) - set(ext_names)
        
        if missing_extensions:
            print(f"❌ Missing required extensions: {', '.join(missing_extensions)}")
            print("Please ensure your PostgreSQL instance supports these extensions")
            return False
        
        print(f"✅ Found extensions: {', '.join(ext_names)}")
        
        # Test a simple query
        result = await conn.fetchval("SELECT COUNT(*) FROM documents")
        print(f"📊 Documents table ready (current count: {result})")
        
        await conn.close()
        print("🎉 Database setup completed successfully!")
        return True
        
    except asyncpg.exceptions.InvalidCatalogNameError:
        print(f"❌ Database does not exist. Please create the database first:")
        print(f"   CREATE DATABASE your_database_name;")
        return False
    except asyncpg.exceptions.InvalidPasswordError:
        print("❌ Invalid database credentials")
        return False
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

async def test_connection_only():
    """Test database connection without executing schema"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        return False
    
    try:
        conn = await asyncpg.connect(database_url)
        result = await conn.fetchval("SELECT 1")
        await conn.close()
        print("✅ Database connection test successful")
        return True
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-only":
        success = asyncio.run(test_connection_only())
    else:
        success = asyncio.run(setup_database())
    
    sys.exit(0 if success else 1)