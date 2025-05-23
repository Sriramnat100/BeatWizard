from supabase import create_client
import os
from dotenv import load_dotenv

def test_connection():
    # Load environment variables
    load_dotenv()
    
    # Initialize Supabase client
    supabase = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_KEY')
    )
    
    try:
        # Try to sign in with your credentials
        response = supabase.auth.sign_in_with_password({
            "email": "sriramnat123@gmail.com",
            "password": "yourpasswordhere"  # Replace with your actual password
        })
        print("Successfully signed in!")
        print(f"User ID: {response.user.id}")
        
        # Try to fetch the user's profile
        profile = supabase.table('profiles').select("*").eq('id', response.user.id).execute()
        print("\nUser Profile:", profile.data)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_connection() 