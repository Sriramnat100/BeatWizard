import 'package:supabase_flutter/supabase_flutter.dart';

// Replace these with your Supabase project credentials
const supabaseUrl = 'https://alncoqxfrhpnmbyzxwwz.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFsbmNvcXhmcmhwbm1ieXp4d3d6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc5NTkzOTYsImV4cCI6MjA2MzUzNTM5Nn0.3EbTCpYBY1xeWOFMFFKNq07nL7qIIA_om3l7kF__CtY';

class SupabaseConfig {
  static SupabaseClient get client => Supabase.instance.client;

  static Future<void> initialize() async {
    await Supabase.initialize(
      url: supabaseUrl,
      anonKey: supabaseAnonKey,
      authOptions: const FlutterAuthClientOptions(
        authFlowType: AuthFlowType.pkce,
      ),
    );
  }
} 