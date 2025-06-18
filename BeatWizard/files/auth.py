import os
from supabase import create_client, Client
from dotenv import load_dotenv

def get_supabase_client() -> Client:
    """Create and return a Supabase client instance"""
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("Missing Supabase credentials. Please check your .env file.")
    
    try:
        return create_client(url, key)
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        raise

def get_current_user(supabase: Client):
    """Get the currently authenticated user"""
    try:
        user = supabase.auth.get_user()
        if user and user.user:
            return user.user
        return None
    except Exception as e:
        print(f"Error getting current user: {e}")
        return None

def sign_in_user(supabase: Client, email: str, password: str):
    """Sign in a user with email and password"""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return response.user if response else None
    except Exception as e:
        print(f"Error signing in: {e}")
        return None

def sign_up_user(supabase: Client, email: str, password: str, username: str = None):
    """Sign up a new user"""
    try:
        signup_data = {
            "email": email,
            "password": password
        }
        if username:
            signup_data["data"] = {"username": username}
            
        response = supabase.auth.sign_up(signup_data)
        return response.user if response else None
    except Exception as e:
        print(f"Error signing up: {e}")
        return None

def sign_out_user(supabase: Client):
    """Sign out the current user"""
    try:
        supabase.auth.sign_out()
        return True
    except Exception as e:
        print(f"Error signing out: {e}")
        return False