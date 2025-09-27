-- Initialize MORPHEUS database
-- This file is run automatically when the PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create morpheus user if not exists (handled by environment variables)
-- Tables will be created by the application on startup