import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'pages/interest_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Supabase.initialize(
  url: 'https://alncoqxfrhpnmbyzxwwz.supabase.co',
  anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFsbmNvcXhmcmhwbm1ieXp4d3d6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc5NTkzOTYsImV4cCI6MjA2MzUzNTM5Nn0.3EbTCpYBY1xeWOFMFFKNq07nL7qIIA_om3l7kF__CtY',
);

  runApp(const InterestApp());
}

class InterestApp extends StatelessWidget {
  const InterestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BeatWizard Interest',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const InterestPage(),
    );
  }
} 