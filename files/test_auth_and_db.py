from auth import Auth
from database import Database
import asyncio

async def test_system():
    auth = Auth()
    db = Database()
    
    try:
        # Test user signup
        test_user = {
            "email": "test@example.com",  # Replace with your email
            "password": "your_test_password",  # Replace with your password
            "username": "testuser"
        }

        # Try to sign in first (if user exists)
        try:
            user = await auth.sign_in(test_user["email"], test_user["password"])
            print("Signed in existing user")
        except Exception:
            # If sign in fails, try to sign up
            user = await auth.sign_up(
                test_user["email"],
                test_user["password"],
                test_user["username"]
            )
            print("Created new user")

        print(f"User ID: {user.id}")

        # Test uploading a beat
        test_beat_data = {
            "title": "Test Beat",
            "description": "A test beat for development",
            "storage_url": "test_url",
            "duration_seconds": 180
        }

        test_audio_features = {
            "tempo": 120.5,
            "rhythm_density": 3.5,
            "beat_consistency": 0.8,
            "groove_strength": 0.7,
            "syncopation_score": 0.6,
            "sub_bass_energy": 0.5,
            "bass_energy": 0.8,
            "bass_to_total_ratio": 0.3,
            "energy_mean": 0.7,
            "energy_std": 0.1,
            "spectral_centroid_mean": 1000,
            "spectral_rolloff_mean": 2000,
            "spectral_bandwidth_mean": 1500,
            "spectral_contrast": 0.6,
            "percussive_energy": 0.7,
            "harmonic_energy": 0.6,
            "percussion_to_harmonic_ratio": 1.2,
            "mfcc_means": [1.0, 2.0, 3.0, 4.0, 5.0]
        }

        # Upload test beat
        beat_id = await db.upload_beat_analysis(
            user_id=user.id,
            beat_data=test_beat_data,
            audio_features=test_audio_features
        )
        
        print(f"Successfully uploaded test beat with ID: {beat_id}")

        # Test getting recommendations
        recs = await db.get_beat_recommendations(beat_id)
        print(f"Got {len(recs)} recommendations")

        # Test interaction
        await db.add_interaction(user.id, beat_id, "like")
        print("Successfully recorded test interaction")

    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(test_system())