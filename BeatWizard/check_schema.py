#!/usr/bin/env python3

import os
from supabase import create_client, Client

# Supabase configuration
url = "https://alncoqxfrhpnmbyzxwwz.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFsbmNvcXhmcmhwbm1ieXp4d3d6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI4MTE2NDcsImV4cCI6MjA0ODM4NzY0N30.QHUcEQDJFnqVU5BVApF2dMDJgMvJMx-VJDMVpYmbOhY"

supabase: Client = create_client(url, key)

# Check current beats table schema
try:
    print("Checking current beats table...")
    
    # Try to get table info
    result = supabase.rpc('get_table_columns', {'table_name': 'beats'}).execute()
    print("Current beats table columns:", result.data)
    
except Exception as e:
    print(f"Could not get table info: {e}")
    print("Let's try a different approach...")

# SQL to create missing columns
missing_columns_sql = """
-- Add missing columns to beats table
ALTER TABLE beats 
ADD COLUMN IF NOT EXISTS bpm INTEGER,
ADD COLUMN IF NOT EXISTS key_signature TEXT,
ADD COLUMN IF NOT EXISTS duration_seconds INTEGER,
ADD COLUMN IF NOT EXISTS is_public BOOLEAN DEFAULT true;

-- Verify the table structure
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'beats' 
ORDER BY ordinal_position;
"""

print("\nSQL to fix missing columns:")
print(missing_columns_sql)

print("\n" + "="*50)
print("NEXT STEPS:")
print("1. Go to your Supabase Dashboard")
print("2. Navigate to SQL Editor")
print("3. Run the SQL above to add missing columns")
print("4. Try uploading again!")
print("="*50) 