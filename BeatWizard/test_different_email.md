# Debug Steps for Auth Error

## Try these steps:

1. **Use a completely different email** (not s@gmail.com)
   - Try something like: testuser123@gmail.com
   - Make sure it's an email that has NEVER been used before

2. **Check Supabase Auth Settings:**
   - Go to your Supabase dashboard
   - Authentication > Settings
   - Make sure "Enable email confirmations" is turned OFF for testing
   - Make sure "Disable new user signups" is turned OFF

3. **Clear any existing data:**
   - Go to Authentication > Users
   - Delete any existing users with the same email

4. **If still not working, try this simplified test:**
   ```dart
   // Just test auth without profile creation
   final authResponse = await _client.auth.signUp(
     email: 'uniquetest123@gmail.com',
     password: 'testpass123',
   );
   print('Auth response: ${authResponse.user?.id}');
   ```

The error suggests the auth.users table itself has an issue, not our profiles table. 