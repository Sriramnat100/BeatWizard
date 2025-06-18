from auth import get_supabase_client, sign_in_user

def setup_storage():
    """Set up Supabase storage bucket and policies"""
    print("Setting up Supabase storage...")
    try:
        # Get Supabase client and authenticate
        supabase = get_supabase_client()
        
        print("\n1. Authenticating...")
        user = sign_in_user(supabase, "sriramnat123@gmail.com", "Sriram123")
        if not user:
            print("✗ Authentication failed")
            return False
        print(f"✓ Authenticated as {user.email}")
        
        print("\n2. Setting up storage bucket...")
        # Create the beats bucket if it doesn't exist
        try:
            supabase.storage.create_bucket("beats", options={
                "public": True
            })
            print("✓ Created 'beats' storage bucket")
        except Exception as e:
            if 'Duplicate' in str(e):
                print("✓ 'beats' bucket already exists")
            else:
                raise e
        
        # Verify bucket exists
        print("\n3. Verifying storage configuration...")
        buckets = supabase.storage.list_buckets()
        beats_bucket = next((b for b in buckets if b.name == 'beats'), None)
        
        if beats_bucket:
            print("✓ Storage bucket configured successfully")
            
            # List files in bucket to verify access
            try:
                supabase.storage.from_('beats').list()
                print("✓ Storage access verified")
                return True
            except Exception as e:
                print(f"✗ Storage access failed: {e}")
                return False
        else:
            print("✗ Failed to verify bucket configuration")
            return False
            
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        return False

if __name__ == "__main__":
    print("=== BeatWizard Storage Setup ===")
    if setup_storage():
        print("\n✅ Storage setup completed successfully!")
    else:
        print("\n❌ Storage setup failed. Please check the errors above.") 