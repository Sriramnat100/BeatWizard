-- First, drop existing policies
DROP POLICY IF EXISTS "Users can view their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can create their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can update their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can delete their own profile" ON profiles;
DROP POLICY IF EXISTS "Enable insert for authenticated users only" ON profiles;

-- Enable RLS
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Allow initial profile creation during signup
CREATE POLICY "Enable insert for registration"
ON profiles
FOR INSERT
WITH CHECK (true);

-- Policy for SELECT operations
CREATE POLICY "Users can view their own profile"
ON profiles
FOR SELECT
USING (auth.uid() = id);

-- Policy for UPDATE operations
CREATE POLICY "Users can update their own profile"
ON profiles
FOR UPDATE USING (auth.uid() = id)
WITH CHECK (auth.uid() = id);

-- Policy for DELETE operations
CREATE POLICY "Users can delete their own profile"
ON profiles
FOR DELETE
USING (auth.uid() = id);

-- Grant necessary permissions
GRANT ALL ON profiles TO authenticated;
GRANT ALL ON profiles TO service_role;

-- Trust the service role
ALTER TABLE profiles FORCE ROW LEVEL SECURITY; 