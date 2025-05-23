import os
from supabase import create_client, Client
from dotenv import load_dotenv

class Database:
    def __init__(self):
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Missing Supabase credentials in .env file")
        self.supabase: Client = create_client(url, key)
    
    def upload_beat_analysis(self, user_id: str, beat_data: dict, audio_features: dict):
        """Upload beat analysis to Supabase"""
        try:
            print(f"Attempting to insert beat with user_id: {user_id}")
            
            # Create a new dictionary with all required fields
            complete_beat_data = {
                "user_id": user_id,
                "title": beat_data["title"],
                "description": beat_data["description"],
                "storage_url": beat_data["storage_url"],
                "bpm": beat_data.get("bpm", audio_features.get("tempo", 0.0)),
                "key_signature": beat_data.get("key_signature", "C"),
                "duration_seconds": beat_data["duration_seconds"],
                "is_public": beat_data.get("is_public", True)
            }
            
            # Debug print
            print("Inserting beat with data:", complete_beat_data)
            
            # First insert the beat
            beat_response = self.supabase.table('beats').insert(complete_beat_data).execute()
            
            if not beat_response.data:
                raise Exception("Failed to insert beat")
            
            beat_id = beat_response.data[0]['id']
            print(f"Successfully created beat with ID: {beat_id}")
            
            # Then insert the audio features
            feature_response = self.supabase.table('audio_features').insert({
                "beat_id": beat_id,
                **audio_features
            }).execute()
            
            if not feature_response.data:
                # Cleanup the beat if feature insertion fails
                self.supabase.table('beats').delete().eq('id', beat_id).execute()
                raise Exception("Failed to insert audio features")
            
            return beat_id
            
        except Exception as e:
            print(f"Database error: {e}")
            return None

    def get_beat_recommendations(self, beat_id: str, limit: int = 5):
        """Get similar beats based on audio features"""
        try:
            # First get the target beat's features
            target = self.supabase.table('audio_features').select('*').eq('beat_id', beat_id).execute()
            
            if not target.data:
                raise ValueError(f"No beat found with id {beat_id}")
            
            # Get all other beats with their features
            beats = self.supabase.table('audio_features').select(
                'beat_id',
                'tempo',
                'rhythm_density',
                'beat_consistency',
                'groove_strength',
                'bass_energy',
                'average_energy'
            ).neq('beat_id', beat_id).execute()
            
            return beats.data[:limit]
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            raise e

    def add_interaction(self, user_id: str, beat_id: str, interaction_type: str):
        """Record a user interaction with a beat (like, play, etc.)"""
        try:
            self.supabase.table('interactions').insert({
                'user_id': user_id,
                'beat_id': beat_id,
                'interaction_type': interaction_type
            }).execute()
        except Exception as e:
            print(f"Error recording interaction: {e}")
            raise e