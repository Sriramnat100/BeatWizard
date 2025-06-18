#!/usr/bin/env python3

from supabase import create_client

# Supabase configuration
url = "https://alncoqxfrhpnmbyzxwwz.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFsbmNvcXhmcmhwbm1ieXp4d3d6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI4MTE2NDcsImV4cCI6MjA0ODM4NzY0N30.QHUcEQDJFnqVU5BVApF2dMDJgMvJMx-VJDMVpYmbOhY"

def create_profile():
    supabase = create_client(url, key)
    
    # Login
    auth_response = supabase.auth.sign_in_with_password({
        "email": "sriramnat123@gmail.com",
        "password": "Sriram123"
    })
    
    user_id = auth_response.user.id
    print(f"User ID: {user_id}")
    
    # Create profile
    profile_data = {
        'id': user_id,
        'username': 'sriram_beats'
    }
    
    try:
        result = supabase.from('profiles').insert(profile_data)
        print("✅ Profile created successfully!")
        print("Now try running beat_uploader.py again")
    except Exception as e:
        print(f"Profile creation error: {e}")

if __name__ == "__main__":
    create_profile() 