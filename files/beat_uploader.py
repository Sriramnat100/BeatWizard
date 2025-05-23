from beat_analyzer import AudioFeatureExtractor, SentimentAnalyzer
from database import Database
from auth import Auth
import os
from pathlib import Path
import librosa
import sys

class BeatUploader:
    def __init__(self):
        try:
            self.feature_extractor = AudioFeatureExtractor()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.db = Database()
            self.auth = Auth()
            self.user_id = None
        except ValueError as e:
            print(f"Initialization error: {e}")
            sys.exit(1)
    
    def login(self, email: str, password: str):
        """Login to Supabase"""
        try:
            user = self.auth.sign_in(email, password)
            if not user or not user.id:
                print("Invalid user response from authentication")
                return False
            self.user_id = user.id
            print(f"Logged in as user: {self.user_id}")
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False
    
    def upload_beat(self, file_path: str, title: str = None, description: str = None):
        """Analyze and upload a beat"""
        if not self.user_id:
            print("Not logged in. Please login first.")
            return False
        
        try:
            # Extract features
            print(f"Extracting features from {file_path}...")
            features = self.feature_extractor.extract_features(file_path)
            if isinstance(features, str):  # Error message
                print(f"Feature extraction failed: {features}")
                return False
            
            # Get file info
            file_info = Path(file_path)
            if not title:
                title = file_info.stem
            
            print(f"Analyzing {title}...")
            
            # Load audio file to get duration
            y, sr = librosa.load(file_path)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Prepare beat data with all required fields
            beat_data = {
                "title": title,
                "description": description or "",
                "storage_url": str(file_info.absolute()),  # Store absolute path
                "duration_seconds": float(duration),
                "is_public": True,  # Default to public
                "bpm": features.get("tempo", 0.0),  # Use tempo as BPM
                "key_signature": "C",  # Default key signature
            }
            
            # If description provided, analyze sentiment
            if description:
                sentiment = self.sentiment_analyzer.analyze_text(description)
                # Add sentiment tags later
            
            # Upload to database
            print("Uploading to database...")
            beat_id = self.db.upload_beat_analysis(
                user_id=self.user_id,
                beat_data=beat_data,
                audio_features=features
            )
            
            if beat_id:
                print(f"Successfully uploaded beat: {title} (ID: {beat_id})")
                return beat_id
            else:
                print("Failed to get beat ID from database")
                return False
            
        except Exception as e:
            print(f"Error uploading beat: {e}")
            return False

def main():
    uploader = BeatUploader()
    
    # Login
    print("Logging in...")
    success = uploader.login("sriramnat123@gmail.com", "yourpasswordhere")
    if not success:
        print("Login failed. Exiting.")
        return
    
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
            str(audio_file),
            title=audio_file.stem,
            description=description
        )
        
        if beat_id:
            print(f"Successfully uploaded {audio_file.name}")
        else:
            print(f"Failed to upload {audio_file.name}")

if __name__ == "__main__":
    main() 