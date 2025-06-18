#!/usr/bin/env python3

# SQL to completely recreate the beats and audio_features tables with proper schema

recreate_schema_sql = """
-- Drop existing tables (since they have wrong structure)
DROP TABLE IF EXISTS audio_features CASCADE;
DROP TABLE IF EXISTS beats CASCADE;

-- Create beats table (basic metadata only)
CREATE TABLE beats (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    storage_url TEXT NOT NULL,
    bpm INTEGER,
    key_signature TEXT,
    duration_seconds INTEGER,
    is_public BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create audio_features table (all the analysis data)
CREATE TABLE audio_features (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    beat_id UUID NOT NULL REFERENCES beats(id) ON DELETE CASCADE,
    tempo REAL,
    rhythm_density REAL,
    beat_consistency REAL,
    syncopation_score REAL,
    groove_strength REAL,
    average_beat_strength REAL,
    energy_mean REAL,
    energy_std REAL,
    sub_bass_energy REAL,
    bass_energy REAL,
    bass_to_total_ratio REAL,
    spectral_centroid REAL,
    spectral_rolloff REAL,
    spectral_bandwidth REAL,
    spectral_contrast REAL,
    percussive_energy REAL,
    harmonic_energy REAL,
    percussion_harmonic_ratio REAL,
    mfcc_1 REAL,
    mfcc_2 REAL,
    mfcc_3 REAL,
    mfcc_4 REAL,
    mfcc_5 REAL,
    kick_energy REAL,
    snare_energy REAL,
    hihat_energy REAL,
    rhythmic_regularity REAL,
    section_changes REAL,
    stereo_width REAL,
    kick_to_snare_ratio REAL,
    hihat_to_kick_ratio REAL,
    duration REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_beats_user_id ON beats(user_id);
CREATE INDEX idx_beats_created_at ON beats(created_at);
CREATE INDEX idx_beats_is_public ON beats(is_public);
CREATE INDEX idx_audio_features_beat_id ON audio_features(beat_id);

-- Enable RLS (Row Level Security)
ALTER TABLE beats ENABLE ROW LEVEL SECURITY;
ALTER TABLE audio_features ENABLE ROW LEVEL SECURITY;

-- RLS Policies for beats table
CREATE POLICY "Users can view public beats" ON beats FOR SELECT USING (is_public = true OR auth.uid() = user_id);
CREATE POLICY "Users can insert their own beats" ON beats FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own beats" ON beats FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own beats" ON beats FOR DELETE USING (auth.uid() = user_id);

-- RLS Policies for audio_features table
CREATE POLICY "Users can view audio features for accessible beats" ON audio_features FOR SELECT USING (
    EXISTS (
        SELECT 1 FROM beats 
        WHERE beats.id = audio_features.beat_id 
        AND (beats.is_public = true OR beats.user_id = auth.uid())
    )
);
CREATE POLICY "Users can insert audio features for their beats" ON audio_features FOR INSERT WITH CHECK (
    EXISTS (
        SELECT 1 FROM beats 
        WHERE beats.id = audio_features.beat_id 
        AND beats.user_id = auth.uid()
    )
);
CREATE POLICY "Users can update audio features for their beats" ON audio_features FOR UPDATE USING (
    EXISTS (
        SELECT 1 FROM beats 
        WHERE beats.id = audio_features.beat_id 
        AND beats.user_id = auth.uid()
    )
);
CREATE POLICY "Users can delete audio features for their beats" ON audio_features FOR DELETE USING (
    EXISTS (
        SELECT 1 FROM beats 
        WHERE beats.id = audio_features.beat_id 
        AND beats.user_id = auth.uid()
    )
);

-- Verify the tables were created correctly
SELECT 'beats table columns:' as info;
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'beats' 
ORDER BY ordinal_position;

SELECT 'audio_features table columns:' as info;
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'audio_features' 
ORDER BY ordinal_position;
"""

print("COMPLETE SCHEMA RECREATION SQL:")
print("="*60)
print(recreate_schema_sql)
print("="*60)
print("\nNEXT STEPS:")
print("1. Go to Supabase Dashboard → SQL Editor")
print("2. Run the SQL above to recreate both tables properly")
print("3. This will give you:")
print("   - beats table: basic metadata (title, description, storage_url, etc.)")
print("   - audio_features table: all analysis data (tempo, energy, mfcc, etc.)")
print("4. Try uploading again!")
print("="*60) 