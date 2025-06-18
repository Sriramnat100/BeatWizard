-- Simple approach: Drop and recreate profiles table without foreign key constraint
DROP TABLE IF EXISTS public.profiles CASCADE;

-- Create simple profiles table (no foreign key for now)
CREATE TABLE public.profiles (
    id UUID PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    bio TEXT,
    avatar_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Simple policy to allow profile creation
CREATE POLICY "Enable insert for registration"
ON profiles
FOR INSERT
WITH CHECK (true);

-- Policy for users to view their own profile
CREATE POLICY "Users can view their own profile"
ON profiles
FOR SELECT
USING (auth.uid() = id);

-- Policy for users to update their own profile
CREATE POLICY "Users can update their own profile"
ON profiles
FOR UPDATE USING (auth.uid() = id)
WITH CHECK (auth.uid() = id);

-- Grant permissions
GRANT ALL ON profiles TO authenticated;
GRANT ALL ON profiles TO service_role; 