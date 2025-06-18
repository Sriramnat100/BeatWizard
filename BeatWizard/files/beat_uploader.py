from beat_analyzer import AudioFeatureExtractor, SentimentAnalyzer
from database import Database
from auth import get_supabase_client, sign_in_user
import os
from pathlib import Path
import librosa
import sys
import numpy as np
from typing import Dict, Any, Tuple
from supabase import Client
import uuid

class BeatUploader:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.database = Database(supabase_client)
        
    def upload_beat(self, file_path: str, user_id: str, title: str, description: str = "", is_public: bool = True) -> str:
        """Upload a beat file and analyze its features"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Beat file not found: {file_path}")
            
            # Extract audio features
            audio_features = self._extract_audio_features(file_path)
            
            # Prepare beat metadata
            beat_data = {
                "title": title,
                "description": description,
                "storage_url": str(Path(file_path).absolute()),  # Store absolute path
                "duration_seconds": audio_features.get("duration", 0),
                "is_public": is_public
            }
            
            # Upload to database
            beat_id = self.database.upload_beat_analysis(user_id, beat_data, audio_features)
            
            if not beat_id:
                raise Exception("Failed to upload beat analysis to database")
            
            return beat_id
            
        except Exception as e:
            print(f"Error uploading beat: {e}")
            return None
    
    def _extract_audio_features(self, file_path: str) -> Dict[str, Any]:
        """Extract audio features from the beat file"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path)
            
            # Basic features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Energy features
            energy = librosa.feature.rms(y=y)[0]
            energy_mean = float(np.mean(energy))
            energy_std = float(np.std(energy))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # Bass and sub-bass energy
            spec = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            sub_bass_mask = freqs <= 60
            bass_mask = (freqs > 60) & (freqs <= 250)
            
            sub_bass_energy = float(np.mean(spec[sub_bass_mask].sum(axis=0)))
            bass_energy = float(np.mean(spec[bass_mask].sum(axis=0)))
            total_energy = float(np.mean(spec.sum(axis=0)))
            
            # Rhythm features
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_est = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfcc, axis=1)
            
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Percussion component analysis
            percussive_energy = float(np.mean(librosa.feature.rms(y=y_percussive)[0]))
            harmonic_energy = float(np.mean(librosa.feature.rms(y=y_harmonic)[0]))
            
            # Extract drum components
            kick_band = (freqs >= 50) & (freqs <= 100)
            snare_band = (freqs >= 200) & (freqs <= 400)
            hihat_band = (freqs >= 10000) & (freqs <= 15000)
            
            kick_energy = float(np.mean(spec[kick_band].sum(axis=0)))
            snare_energy = float(np.mean(spec[snare_band].sum(axis=0)))
            hihat_energy = float(np.mean(spec[hihat_band].sum(axis=0)))
            
            # Calculate rhythm density from onset envelope
            rhythm_density = float(np.mean(onset_env))
            
            # Calculate beat consistency
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                beat_consistency = float(1.0 / (np.std(beat_intervals) + 1e-6))
            else:
                beat_consistency = 0.0
            
            return {
                "tempo": float(tempo),
                "duration": float(duration),
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "spectral_centroid": float(np.mean(spectral_centroids)),
                "spectral_rolloff": float(np.mean(spectral_rolloff)),
                "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
                "sub_bass_energy": sub_bass_energy,
                "bass_energy": bass_energy,
                "bass_to_total_ratio": float(bass_energy / total_energy if total_energy > 0 else 0),
                "mfcc_1": float(mfcc_means[0]),
                "mfcc_2": float(mfcc_means[1]),
                "mfcc_3": float(mfcc_means[2]),
                "mfcc_4": float(mfcc_means[3]),
                "mfcc_5": float(mfcc_means[4]),
                "percussive_energy": percussive_energy,
                "harmonic_energy": harmonic_energy,
                "percussion_harmonic_ratio": float(percussive_energy / harmonic_energy if harmonic_energy > 0 else 0),
                "rhythm_density": rhythm_density,
                "beat_consistency": beat_consistency,
                "kick_energy": kick_energy,
                "snare_energy": snare_energy,
                "hihat_energy": hihat_energy,
                "kick_to_snare_ratio": float(kick_energy / snare_energy if snare_energy > 0 else 0),
                "hihat_to_kick_ratio": float(hihat_energy / kick_energy if kick_energy > 0 else 0)
            }
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {}

def main():
    try:
        # Initialize Supabase client
        supabase = get_supabase_client()
        
        # Login
        print("Logging in...")
        user = sign_in_user(supabase, "sriramnat123@gmail.com", "Sriram123")
        if not user:
            print("Login failed. Exiting.")
            return
            
        # Ensure profile exists
        print(f"Checking/creating profile for user {user.id}...")
        try:
            # Try to get existing profile
            profile_check = supabase.from('profiles').select('*').eq('id', user.id).execute()
            
            if not profile_check.data:
                # Create profile if it doesn't exist
                print("Creating profile...")
                profile_data = {
                    'id': user.id,
                    'username': 'sriram_beats',  # You can change this
                }
                supabase.from('profiles').insert(profile_data).execute()
                print("✅ Profile created successfully!")
            else:
                print("✅ Profile already exists!")
                
        except Exception as e:
            print(f"Error with profile: {e}")
            return
            
        # Initialize uploader with authenticated client
        uploader = BeatUploader(supabase)
        
        # Process all beats in audio_samples directory
        audio_dir = Path("audio_samples")
        if not audio_dir.exists():
            print(f"Error: {audio_dir} directory not found")
            return
            
        mp3_files = list(audio_dir.glob("*.mp3"))
        if not mp3_files:
            print(f"No MP3 files found in {audio_dir}")
            return
        
        print(f"\nFound {len(mp3_files)} MP3 files to process")
        
        for audio_file in mp3_files:
            print(f"\nProcessing {audio_file.name}...")
            
            # Get description from user
            description = input(f"Enter description for {audio_file.name} (or press Enter to skip): ")
            
            # Upload beat
            beat_id = uploader.upload_beat(
                file_path=str(audio_file),
                user_id=user.id,
                title=audio_file.stem,
                description=description
            )
            
            if beat_id:
                print(f"Successfully uploaded {audio_file.name} with ID: {beat_id}")
            else:
                print(f"Failed to upload {audio_file.name}")

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main() 