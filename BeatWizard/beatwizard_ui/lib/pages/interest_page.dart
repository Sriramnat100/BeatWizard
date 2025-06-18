import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'dart:math' as math;
import 'package:google_fonts/google_fonts.dart';

class InterestPage extends StatefulWidget {
  const InterestPage({super.key});

  @override
  State<InterestPage> createState() => _InterestPageState();
}

const String kPlayfulFont = 'Comic Sans MS';

class _InterestPageState extends State<InterestPage> with SingleTickerProviderStateMixin {
  final _emailController = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  String? _errorMessage;
  String _selectedUserType = 'artist'; // Default value

  late AnimationController _colorController;
  late Animation<Color?> _colorAnimation;
  final List<Color> _colors = [
    Color(0xFF0072b1), // LinkedIn blue
    Color(0xFF6C63FF), // purple
    Color(0xFF00B8A9), // teal
    Color(0xFFFC5185), // pink
    Color(0xFF43E97B), // green
    Color(0xFF38A1F3), // Twitter blue
  ];
  final String _brand = 'BeatBoard';

  @override
  void initState() {
    super.initState();
    _colorController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 8),
    )..repeat();
    _colorAnimation = _colorController.drive(
      TweenSequence<Color?>([
        for (int i = 0; i < _colors.length; i++)
          TweenSequenceItem(
            tween: ColorTween(
              begin: _colors[i],
              end: _colors[(i + 1) % _colors.length],
            ),
            weight: 1,
          ),
      ]),
    );
  }

  @override
  void dispose() {
    _colorController.dispose();
    _emailController.dispose();
    super.dispose();
  }

  Future<void> _submitEmail() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final supabase = Supabase.instance.client;
      final data = {
        'email': _emailController.text.trim(),
        'user_type': _selectedUserType,
      };
      print('Attempting to insert data: $data'); // Debug log
      
      final response = await supabase.from('interest_form').insert(data);
      print('Insert response: $response'); // Debug log

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Thanks for your interest! We\'ll be in touch soon.'),
            backgroundColor: Colors.green,
          ),
        );
        _emailController.clear();
      }
    } catch (e) {
      print('Error during submission: $e'); // Debug log
      setState(() {
        _errorMessage = 'Failed to submit: ${e.toString()}';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFFF3F2EF), // LinkedIn light gray
              Color(0xFFE7E9EB), // LinkedIn lighter gray
            ],
          ),
        ),
        child: Stack(
          children: [
            // Main content
            SafeArea(
              child: Center(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 32.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      AnimatedBuilder(
                        animation: _colorController,
                        builder: (context, child) {
                          return Text(
                            _brand,
                            textAlign: TextAlign.center,
                            style: GoogleFonts.montserrat(
                              fontSize: 180,
                              fontWeight: FontWeight.w800,
                              color: _colorAnimation.value,
                              letterSpacing: 2.5,
                              shadows: [
                                Shadow(
                                  color: _colorAnimation.value!.withOpacity(0.25),
                                  blurRadius: 16,
                                  offset: const Offset(0, 8),
                                ),
                              ],
                            ),
                          );
                        },
                      ),
                      const SizedBox(height: 32),
                      Container(
                        constraints: const BoxConstraints(maxWidth: 560),
                        child: Card(
                          elevation: 16,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(18),
                          ),
                          shadowColor: Colors.black.withOpacity(0.12),
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 48.0, vertical: 48.0),
                            child: Form(
                              key: _formKey,
                              child: Column(
                                mainAxisSize: MainAxisSize.min,
                                crossAxisAlignment: CrossAxisAlignment.stretch,
                                children: [
                                  Text(
                                    'Join the Waitlist',
                                    textAlign: TextAlign.center,
                                    style: GoogleFonts.montserrat(
                                      fontSize: 36,
                                      fontWeight: FontWeight.w800,
                                      color: Color(0xFF0A2342),
                                      letterSpacing: 0.2,
                                    ),
                                  ),
                                  const SizedBox(height: 14),
                                  Text(
                                    'Be the first to know when we launch BeatBoard.',
                                    textAlign: TextAlign.center,
                                    style: GoogleFonts.montserrat(
                                      fontSize: 20,
                                      color: Color(0xFF3B3B3B),
                                      fontWeight: FontWeight.w800,
                                    ),
                                  ),
                                  const SizedBox(height: 28),
                                  TextFormField(
                                    controller: _emailController,
                                    style: GoogleFonts.montserrat(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w800,
                                    ),
                                    decoration: InputDecoration(
                                      labelText: 'Email',
                                      hintText: 'Enter your email',
                                      prefixIcon: Icon(
                                        Icons.alternate_email,
                                        color: Color(0xFF0072b1),
                                        size: 24,
                                      ),
                                      labelStyle: GoogleFonts.montserrat(
                                        fontWeight: FontWeight.w800,
                                        color: Color(0xFF0A2342),
                                      ),
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(8),
                                        borderSide: const BorderSide(
                                          color: Color(0xFFE0E3E7),
                                          width: 1.5,
                                        ),
                                      ),
                                      enabledBorder: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(8),
                                        borderSide: const BorderSide(
                                          color: Color(0xFFE0E3E7),
                                          width: 1.5,
                                        ),
                                      ),
                                      focusedBorder: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(8),
                                        borderSide: const BorderSide(
                                          color: Color(0xFF0072b1),
                                          width: 2,
                                        ),
                                      ),
                                      filled: true,
                                      fillColor: Colors.grey.shade50,
                                      contentPadding: const EdgeInsets.symmetric(vertical: 18, horizontal: 18),
                                    ),
                                    keyboardType: TextInputType.emailAddress,
                                    validator: (value) {
                                      if (value == null || value.isEmpty) {
                                        return 'Please enter your email';
                                      }
                                      if (!RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$')
                                          .hasMatch(value)) {
                                        return 'Please enter a valid email';
                                      }
                                      return null;
                                    },
                                  ),
                                  const SizedBox(height: 22),
                                  DropdownButtonFormField<String>(
                                    value: _selectedUserType,
                                    decoration: InputDecoration(
                                      labelText: 'I am a',
                                      prefixIcon: const Icon(Icons.person, color: Color(0xFF0072b1)),
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(8),
                                        borderSide: const BorderSide(
                                          color: Color(0xFFE0E3E7),
                                          width: 1.5,
                                        ),
                                      ),
                                      filled: true,
                                      fillColor: Colors.grey.shade50,
                                    ),
                                    items: [
                                      DropdownMenuItem(
                                        value: 'artist',
                                        child: Text('Artist', style: GoogleFonts.montserrat(fontWeight: FontWeight.w800)),
                                      ),
                                      DropdownMenuItem(
                                        value: 'producer',
                                        child: Text('Producer', style: GoogleFonts.montserrat(fontWeight: FontWeight.w800)),
                                      ),
                                    ],
                                    onChanged: (value) {
                                      if (value != null) {
                                        setState(() {
                                          _selectedUserType = value;
                                        });
                                      }
                                    },
                                  ),
                                  if (_errorMessage != null) ...[
                                    const SizedBox(height: 16),
                                    Text(
                                      _errorMessage!,
                                      style: GoogleFonts.montserrat(
                                        color: Colors.red,
                                        fontSize: 14,
                                        fontWeight: FontWeight.w800,
                                      ),
                                    ),
                                  ],
                                  const SizedBox(height: 32),
                                  MouseRegion(
                                    cursor: SystemMouseCursors.click,
                                    child: AnimatedContainer(
                                      duration: const Duration(milliseconds: 150),
                                      curve: Curves.easeInOut,
                                      decoration: BoxDecoration(
                                        borderRadius: BorderRadius.circular(8),
                                        boxShadow: [
                                          BoxShadow(
                                            color: Colors.blue.shade200.withOpacity(0.25),
                                            blurRadius: 16,
                                            offset: const Offset(0, 6),
                                          ),
                                        ],
                                      ),
                                      child: SizedBox(
                                        width: double.infinity,
                                        height: 52,
                                        child: ElevatedButton(
                                          onPressed: _isLoading ? null : _submitEmail,
                                          style: ElevatedButton.styleFrom(
                                            backgroundColor: const Color(0xFF0072b1),
                                            foregroundColor: Colors.white,
                                            textStyle: GoogleFonts.montserrat(
                                              fontWeight: FontWeight.w800,
                                              fontSize: 18,
                                            ),
                                            shape: RoundedRectangleBorder(
                                              borderRadius: BorderRadius.circular(8),
                                            ),
                                            elevation: 0,
                                            shadowColor: Colors.transparent,
                                          ),
                                          child: _isLoading
                                              ? const SizedBox(
                                                  height: 22,
                                                  width: 22,
                                                  child: CircularProgressIndicator(
                                                    strokeWidth: 2,
                                                    valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                                  ),
                                                )
                                              : Text('Submit', style: GoogleFonts.montserrat(fontWeight: FontWeight.w800)),
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
} 
