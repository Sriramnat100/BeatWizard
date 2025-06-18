import 'package:supabase_flutter/supabase_flutter.dart';
import 'dart:typed_data';
import 'supabase_config.dart';

class UserService {
  final SupabaseClient _client = SupabaseConfig.client;

  Future<User?> getCurrentUser() async {
    return _client.auth.currentUser;
  }

  Future<void> signUp({
    required String email,
    required String password,
    required String username,
    String? bio,
    Uint8List? avatarBytes,
    String? avatarFileName,
  }) async {
    try {
      // First check if username already exists
      final existingProfiles = await _client
          .from('profiles')
          .select()
          .eq('username', username);

      if (existingProfiles != null && existingProfiles.isNotEmpty) {
        throw 'Username already taken';
      }

      // Create auth user
      final authResponse = await _client.auth.signUp(
        email: email,
        password: password,
        data: {'username': username},
      );

      if (authResponse.user == null) {
        throw 'Failed to create user account';
      }

      String? avatarUrl;
      
      // Upload avatar if provided
      if (avatarBytes != null && avatarFileName != null) {
        avatarUrl = await _uploadAvatarBytes(
          authResponse.user!.id, 
          avatarBytes, 
          avatarFileName
        );
      }

      // Create profile in the profiles table
      await _client.from('profiles').insert({
        'id': authResponse.user!.id,
        'username': username,
        'email': email,
        'bio': bio,
        'avatar_url': avatarUrl,
        'created_at': DateTime.now().toIso8601String(),
      });

    } catch (e) {
      // If any error occurs, ensure we clean up
      if (e is! String) { // Only rethrow if it's not our custom error
        throw 'An unexpected error occurred: $e';
      }
      rethrow;
    }
  }

  Future<String?> _uploadAvatarBytes(String userId, Uint8List bytes, String fileName) async {
    try {
      final fileExt = fileName.split('.').last.toLowerCase();
      final uniqueFileName = '${DateTime.now().millisecondsSinceEpoch}_avatar.$fileExt';
      final filePath = '$userId/$uniqueFileName';

      await _client.storage
          .from('avatars')
          .uploadBinary(filePath, bytes);

      return _client.storage
          .from('avatars')
          .getPublicUrl(filePath);
    } catch (e) {
      print('Failed to upload avatar: $e');
      return null; // Don't fail profile creation if avatar upload fails
    }
  }

  Future<void> signIn({
    required String email,
    required String password,
  }) async {
    try {
      final response = await _client.auth.signInWithPassword(
        email: email,
        password: password,
      );
      
      if (response.user == null) {
        throw 'Failed to sign in';
      }
    } catch (e) {
      throw 'Invalid email or password';
    }
  }

  Future<void> signOut() async {
    await _client.auth.signOut();
  }

  Future<void> updateProfile({
    required String userId,
    String? username,
    String? bio,
    String? avatarUrl,
  }) async {
    try {
      final updates = <String, dynamic>{};
      if (username != null) updates['username'] = username;
      if (bio != null) updates['bio'] = bio;
      if (avatarUrl != null) updates['avatar_url'] = avatarUrl;

      await _client
          .from('profiles')
          .update(updates)
          .eq('id', userId);

    } catch (e) {
      throw 'Failed to update profile: $e';
    }
  }

  Future<String?> uploadProfilePicture(String filePath) async {
    try {
      final userId = _client.auth.currentUser?.id;
      if (userId == null) return null;

      final fileExt = filePath.split('.').last;
      final fileName = 'avatar.$fileExt';
      final storageResponse = await _client.storage
          .from('avatars')
          .upload('$userId/$fileName', filePath);

      return storageResponse;
    } catch (e) {
      throw 'Failed to upload profile picture: $e';
    }
  }

  Future<bool> profileExists(String userId) async {
    try {
      final response = await _client
          .from('profiles')
          .select('id')
          .eq('user_id', userId);

      return response.isNotEmpty;
    } catch (e) {
      print('Error checking profile: $e');
      return false;
    }
  }

  Future<void> createProfile({
    required String userId,
    required String username,
    required String firstName,
    required String lastName,
    String? bio,
  }) async {
    try {
      print('Creating profile for user: $userId');
      print('Username: $username');
      
      await _client.from('profiles').insert({
        'user_id': userId,
        'username': username,
        'first_name': firstName,
        'last_name': lastName,
        'bio': bio ?? '',
        'created_at': DateTime.now().toIso8601String(),
      });

      print('Profile created successfully');
    } catch (e) {
      print('Error creating profile: $e');
      throw 'Failed to create profile: $e';
    }
  }

  Future<Map<String, dynamic>?> getProfile(String userId) async {
    try {
      final response = await _client
          .from('profiles')
          .select()
          .eq('user_id', userId)
          .single();

      return response;
    } catch (e) {
      print('Error getting profile: $e');
      return null;
    }
  }

  Future<Map<String, dynamic>?> getCurrentUserProfile() async {
    try {
      final user = _client.auth.currentUser;
      if (user == null) {
        throw 'No authenticated user';
      }

      final response = await _client
          .from('profiles')
          .select()
          .eq('id', user.id)
          .single();

      return response;
    } catch (e) {
      print('Error getting current user profile: $e');
      throw 'Failed to load profile: ${e.toString()}';
    }
  }
} 