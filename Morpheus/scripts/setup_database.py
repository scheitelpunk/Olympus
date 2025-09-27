#!/usr/bin/env python3
"""Setup MORPHEUS PostgreSQL database."""

import subprocess
import time
import sys
import psycopg2
from psycopg2 import sql

def run_command(cmd, shell=True):
    """Run shell command and return result."""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, check=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_docker():
    """Check if Docker is available."""
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print("❌ Docker is not available. Please install Docker first.")
        return False
    
    print(f"✅ Docker found: {stdout.strip()}")
    return True

def start_database():
    """Start PostgreSQL database with Docker Compose."""
    print("🚀 Starting PostgreSQL database...")
    
    success, stdout, stderr = run_command("docker-compose -f docker/docker-compose.yml up -d postgres")
    
    if not success:
        print(f"❌ Failed to start database: {stderr}")
        return False
    
    print("⏳ Waiting for PostgreSQL to be ready...")
    
    # Wait for database to be ready
    for i in range(30):
        success, stdout, stderr = run_command(
            "docker exec morpheus_db pg_isready -U morpheus_user -d morpheus"
        )
        if success:
            print("✅ PostgreSQL is ready!")
            return True
        
        time.sleep(1)
        print(f"   Waiting... ({i+1}/30)")
    
    print("❌ PostgreSQL failed to start within 30 seconds")
    return False

def test_connection():
    """Test database connection."""
    print("🧪 Testing database connection...")
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="morpheus", 
            user="morpheus_user",
            password="morpheus_pass"
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def create_extensions():
    """Create required database extensions."""
    print("🔧 Creating database extensions...")
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="morpheus",
            user="morpheus_user", 
            password="morpheus_pass"
        )
        
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            print("✅ UUID extension created")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create extensions: {e}")
        return False

def main():
    """Main setup function."""
    print("=== MORPHEUS Database Setup ===\n")
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
    
    # Start database
    if not start_database():
        sys.exit(1)
    
    # Test connection
    if not test_connection():
        sys.exit(1)
    
    # Create extensions
    if not create_extensions():
        sys.exit(1)
    
    print("\n🎉 Database setup complete!")
    print("📊 Access Adminer at: http://localhost:8080")
    print("🗄️  PostgreSQL available at: localhost:5432")
    print("📚 Database: morpheus")
    print("👤 User: morpheus_user")
    
    print("\n💡 Next steps:")
    print("1. Install Python dependencies: pip install -r requirements.txt")
    print("2. Run examples: python -m morpheus.examples.basic_perception")

if __name__ == "__main__":
    main()