from supabase import create_client, Client
import os
from dotenv import load_dotenv

class Auth:
    def __init__(self):
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Missing Supabase credentials in .env file")
        self.supabase: Client = create_client(url, key)
        self.current_user = None

    async def sign_up(self, email: str, password: str, username: str):
        """Register a new user"""
        try:
            # Sign up the user
            auth_response = self.supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "username": username
                    }
                }
            })
            
            return auth_response.user
            
        except Exception as e:
            print(f"Error signing up: {e}")
            raise e

    def sign_in(self, email: str, password: str):
        """Sign in to Supabase"""
        try:
            response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            self.current_user = response.user
            return response.user
        except Exception as e:
            print(f"Authentication error: {e}")
            raise e

    def get_current_user(self):
        """Get the currently signed in user"""
        return self.current_user

    def sign_out(self):
        """Sign out the current user"""
        try:
            self.supabase.auth.sign_out()
            self.current_user = None
        except Exception as e:
            print(f"Error signing out: {e}")
            raise e