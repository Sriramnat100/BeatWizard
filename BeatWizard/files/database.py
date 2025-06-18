from supabase import Client

class Database:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    def upload_beat_analysis(self, user_id: str, beat_data: dict, audio_features: dict):
        """Upload beat analysis to Supabase"""
        try:
            print(f"Attempting to insert beat with user_id: {user_id}")
            
            # Create a new dictionary with all required fields for beats table
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
            
            print("Complete beat data to insert:", complete_beat_data)
            
            # First insert the beat
            beat_response = self.supabase.table('beats').insert(complete_beat_data).execute()
            
            if not beat_response.data:
                raise Exception("Failed to insert beat")
            
            beat_id = beat_response.data[0]['id']
            print(f"Successfully created beat with ID: {beat_id}")
            
            # Add beat_id to audio features
            audio_features["beat_id"] = beat_id
            
            # Remove any features that aren't in our schema
            valid_features = {
                "beat_id", "tempo", "duration", "energy_mean", "energy_std",
                "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
                "sub_bass_energy", "bass_energy", "bass_to_total_ratio",
                "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
                "percussive_energy", "harmonic_energy", "percussion_harmonic_ratio",
                "rhythm_density", "beat_consistency", "kick_energy", "snare_energy",
                "hihat_energy", "kick_to_snare_ratio", "hihat_to_kick_ratio"
            }
            
            # Filter features
            filtered_features = {k: v for k, v in audio_features.items() if k in valid_features}
            
            # Ensure all required fields exist with defaults
            for field in valid_features:
                if field not in filtered_features:
                    filtered_features[field] = 0.0
            
            print("Inserting audio features:", filtered_features)
            
            # Insert audio features
            feature_response = self.supabase.table('audio_features').insert(filtered_features).execute()
            
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
                'energy_mean',
                'bass_energy',
                'rhythm_density',
                'beat_consistency'
            ).neq('beat_id', beat_id).execute()
            
            return beats.data[:limit]
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None

    def add_interaction(self, user_id: str, beat_id: str, interaction_type: str):
        """Record a user interaction with a beat (like, play, etc.)"""
        try:
            response = self.supabase.table('interactions').insert({
                'user_id': user_id,
                'beat_id': beat_id,
                'interaction_type': interaction_type
            }).execute()
            return True
        except Exception as e:
            print(f"Error recording interaction: {e}")
            return False