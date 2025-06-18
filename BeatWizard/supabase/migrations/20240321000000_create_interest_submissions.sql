-- Create interest_submissions table
CREATE TABLE IF NOT EXISTS interest_submissions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_interest_submissions_email ON interest_submissions(email);

-- Add RLS policies
ALTER TABLE interest_submissions ENABLE ROW LEVEL SECURITY;

-- Allow anonymous inserts
CREATE POLICY "Allow anonymous inserts" ON interest_submissions
    FOR INSERT
    TO anon
    WITH CHECK (true);

-- Allow authenticated users to view all submissions
CREATE POLICY "Allow authenticated users to view submissions" ON interest_submissions
    FOR SELECT
    TO authenticated
    USING (true);

-- Add user_type column with check constraint
ALTER TABLE interest_submissions 
ADD COLUMN user_type TEXT NOT NULL DEFAULT 'artist' 
CHECK (user_type IN ('artist', 'producer')); 