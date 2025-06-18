-- Drop all dependent tables in reverse dependency order
DROP TABLE IF EXISTS public.beat_tags CASCADE;
DROP TABLE IF EXISTS public.audio_features CASCADE;
DROP TABLE IF EXISTS public.interactions CASCADE;
DROP TABLE IF EXISTS public.beats CASCADE;
DROP TABLE IF EXISTS public.profiles CASCADE;

-- Create profiles table
CREATE TABLE public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    bio TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recreate beats table
CREATE TABLE public.beats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    file_url TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recreate interactions table
CREATE TABLE public.interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    beat_id UUID REFERENCES public.beats(id) ON DELETE CASCADE,
    interaction_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recreate audio_features table
CREATE TABLE public.audio_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    beat_id UUID REFERENCES public.beats(id) ON DELETE CASCADE,
    tempo FLOAT,
    key_signature TEXT,
    time_signature TEXT,
    loudness FLOAT,
    energy FLOAT,
    danceability FLOAT,
    valence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recreate beat_tags table
CREATE TABLE public.beat_tags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    beat_id UUID REFERENCES public.beats(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Set up Row Level Security (RLS)
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.beats ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audio_features ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.beat_tags ENABLE ROW LEVEL SECURITY;

-- Create indexes
CREATE INDEX profiles_username_idx ON public.profiles(username);
CREATE INDEX profiles_email_idx ON public.profiles(email);
CREATE INDEX beats_user_id_idx ON public.beats(user_id);
CREATE INDEX interactions_user_id_idx ON public.interactions(user_id);
CREATE INDEX interactions_beat_id_idx ON public.interactions(beat_id);
CREATE INDEX audio_features_beat_id_idx ON public.audio_features(beat_id);
CREATE INDEX beat_tags_beat_id_idx ON public.beat_tags(beat_id);
CREATE INDEX beat_tags_user_id_idx ON public.beat_tags(user_id);

-- Set up realtime
ALTER PUBLICATION supabase_realtime ADD TABLE public.profiles;
ALTER PUBLICATION supabase_realtime ADD TABLE public.beats;
ALTER PUBLICATION supabase_realtime ADD TABLE public.interactions;
ALTER PUBLICATION supabase_realtime ADD TABLE public.audio_features;
ALTER PUBLICATION supabase_realtime ADD TABLE public.beat_tags;

-- Handle updated_at for all tables that need it
CREATE TRIGGER handle_updated_at_profiles BEFORE UPDATE ON public.profiles 
  FOR EACH ROW EXECUTE PROCEDURE moddatetime (updated_at);

CREATE TRIGGER handle_updated_at_beats BEFORE UPDATE ON public.beats 
  FOR EACH ROW EXECUTE PROCEDURE moddatetime (updated_at);

CREATE TRIGGER handle_updated_at_interactions BEFORE UPDATE ON public.interactions 
  FOR EACH ROW EXECUTE PROCEDURE moddatetime (updated_at); 