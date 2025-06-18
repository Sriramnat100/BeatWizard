import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:go_router/go_router.dart';

class LandingPage extends StatelessWidget {
  const LandingPage({super.key});

  Widget _buildBeatWizardHeroLogo(BuildContext context) {
    return Container(
      width: 120,
      height: 120,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF8B5CF6),
            const Color(0xFF6366F1),
            const Color(0xFF3B82F6),
          ],
        ),
        borderRadius: BorderRadius.circular(25),
        boxShadow: [
          BoxShadow(
            color: Theme.of(context).colorScheme.primary.withOpacity(0.4),
            blurRadius: 20,
            spreadRadius: 5,
          ),
        ],
      ),
      child: Stack(
        children: [
          // Wizard hat - larger version
          Positioned(
            top: 8,
            left: 20,
            right: 20,
            child: Container(
              height: 50,
              decoration: BoxDecoration(
                color: const Color(0xFF8B4D8E),
                borderRadius: BorderRadius.circular(25),
              ),
            ),
          ),
          // Hat brim
          Positioned(
            top: 45,
            left: 10,
            right: 10,
            child: Container(
              height: 12,
              decoration: BoxDecoration(
                color: const Color(0xFF4A2C4A),
                borderRadius: BorderRadius.circular(6),
              ),
            ),
          ),
          // Stars on hat
          const Positioned(
            top: 20,
            left: 35,
            child: Icon(Icons.star, color: Color(0xFFF59E0B), size: 16),
          ),
          const Positioned(
            top: 15,
            right: 30,
            child: Icon(Icons.star, color: Color(0xFFF59E0B), size: 12),
          ),
          // Face/beard area
          Positioned(
            top: 50,
            left: 25,
            right: 25,
            child: Container(
              height: 35,
              decoration: BoxDecoration(
                color: const Color(0xFFD97706),
                borderRadius: BorderRadius.circular(20),
              ),
            ),
          ),
          // Beard
          Positioned(
            top: 70,
            left: 20,
            right: 20,
            child: Container(
              height: 25,
              decoration: BoxDecoration(
                color: const Color(0xFFF3F4F6),
                borderRadius: BorderRadius.circular(15),
              ),
            ),
          ),
          // Turntables - larger version
          Positioned(
            bottom: 8,
            left: 15,
            child: Container(
              width: 25,
              height: 25,
              decoration: const BoxDecoration(
                color: Color(0xFF374151),
                shape: BoxShape.circle,
              ),
              child: Center(
                child: Container(
                  width: 12,
                  height: 12,
                  decoration: const BoxDecoration(
                    color: Color(0xFFF59E0B),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            bottom: 8,
            right: 15,
            child: Container(
              width: 25,
              height: 25,
              decoration: const BoxDecoration(
                color: Color(0xFF374151),
                shape: BoxShape.circle,
              ),
              child: Center(
                child: Container(
                  width: 12,
                  height: 12,
                  decoration: const BoxDecoration(
                    color: Color(0xFFF59E0B),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            ),
          ),
          // Mixer in center
          Positioned(
            bottom: 12,
            left: 45,
            right: 45,
            child: Container(
              height: 15,
              decoration: BoxDecoration(
                color: const Color(0xFF4B5563),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Container(
                    width: 4,
                    height: 4,
                    decoration: const BoxDecoration(
                      color: Color(0xFF10B981),
                      shape: BoxShape.circle,
                    ),
                  ),
                  Container(
                    width: 4,
                    height: 4,
                    decoration: const BoxDecoration(
                      color: Color(0xFFEF4444),
                      shape: BoxShape.circle,
                    ),
                  ),
                  Container(
                    width: 4,
                    height: 4,
                    decoration: const BoxDecoration(
                      color: Color(0xFF3B82F6),
                      shape: BoxShape.circle,
                    ),
                  ),
                ],
              ),
            ),
          ),
          // Magic sparkles around the logo
          const Positioned(
            top: 10,
            left: 5,
            child: Icon(Icons.auto_awesome, color: Color(0xFFF59E0B), size: 8),
          ),
          const Positioned(
            top: 30,
            right: 5,
            child: Icon(Icons.auto_awesome, color: Color(0xFF8B5CF6), size: 6),
          ),
          const Positioned(
            bottom: 40,
            left: 8,
            child: Icon(Icons.auto_awesome, color: Color(0xFFF59E0B), size: 6),
          ),
          const Positioned(
            bottom: 35,
            right: 10,
            child: Icon(Icons.auto_awesome, color: Color(0xFF8B5CF6), size: 8),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              const Color(0xFF1A1A2E),
              const Color(0xFF16213E),
              const Color(0xFF0F3460),
              Theme.of(context).colorScheme.primary.withOpacity(0.1),
            ],
            stops: const [0.0, 0.3, 0.7, 1.0],
          ),
        ),
        child: Stack(
          children: [
            // Animated background particles
            ...List.generate(20, (index) => Positioned(
              left: (index * 50.0) % MediaQuery.of(context).size.width,
              top: (index * 80.0) % MediaQuery.of(context).size.height,
              child: Container(
                width: 4,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.1),
                  shape: BoxShape.circle,
                ),
              ).animate(onPlay: (controller) => controller.repeat())
                .fadeIn(duration: 2000.ms, delay: (index * 100).ms)
                .then()
                .fadeOut(duration: 2000.ms),
            )),
            
            // Main content
            Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(32.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // Hero Logo Section
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [
                            Colors.white.withOpacity(0.1),
                            Colors.white.withOpacity(0.05),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(30),
                        border: Border.all(
                          color: Colors.white.withOpacity(0.1),
                          width: 1,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                            blurRadius: 30,
                            spreadRadius: 10,
                          ),
                        ],
                      ),
                      child: _buildBeatWizardHeroLogo(context),
                    ).animate().fadeIn(duration: 800.ms).scale(delay: 200.ms),
                    
                    const SizedBox(height: 40),
                    
                    // App Title with stunning typography
                    ShaderMask(
                      shaderCallback: (bounds) => LinearGradient(
                        colors: [
                          Colors.white,
                          Theme.of(context).colorScheme.primary,
                          Colors.white,
                        ],
                      ).createShader(bounds),
                      child: Text(
                        'BeatWizard',
                        style: Theme.of(context).textTheme.displayLarge?.copyWith(
                              color: Colors.white,
                              fontWeight: FontWeight.w900,
                              fontSize: 48,
                              letterSpacing: 2,
                              shadows: [
                                Shadow(
                                  color: Theme.of(context).colorScheme.primary.withOpacity(0.5),
                                  offset: const Offset(0, 2),
                                  blurRadius: 10,
                                ),
                              ],
                            ),
                      ),
                    ).animate().fadeIn(delay: 400.ms).slideY(begin: 0.3),
                    
                    const SizedBox(height: 16),
                    
                    // Subtitle with glass effect
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Colors.white.withOpacity(0.1),
                            Colors.white.withOpacity(0.05),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(25),
                        border: Border.all(
                          color: Colors.white.withOpacity(0.1),
                        ),
                      ),
                      child: Text(
                        '🎵 Discover and Share Your Beats',
                        style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                              color: Colors.white.withOpacity(0.9),
                              fontWeight: FontWeight.w500,
                              fontSize: 18,
                            ),
                      ),
                    ).animate().fadeIn(delay: 600.ms).slideY(begin: 0.3),
                    
                    const SizedBox(height: 60),
                    
                    // Action buttons with stunning design
                    Container(
                      width: 340,
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [
                            Colors.white.withOpacity(0.1),
                            Colors.white.withOpacity(0.05),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(
                          color: Colors.white.withOpacity(0.1),
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.1),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          // Sign In Button - Premium style
                          Container(
                            width: double.infinity,
                            height: 56,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                                colors: [
                                  Theme.of(context).colorScheme.primary,
                                  Theme.of(context).colorScheme.primary.withOpacity(0.8),
                                ],
                              ),
                              borderRadius: BorderRadius.circular(16),
                              boxShadow: [
                                BoxShadow(
                                  color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                                  blurRadius: 15,
                                  spreadRadius: 2,
                                  offset: const Offset(0, 4),
                                ),
                              ],
                            ),
                            child: ElevatedButton(
                              onPressed: () => context.go('/auth/signin'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.transparent,
                                shadowColor: Colors.transparent,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16),
                                ),
                              ),
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Icon(Icons.login, color: Colors.white),
                                  const SizedBox(width: 12),
                                  Text(
                                    'Sign In',
                                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                          color: Colors.white,
                                          fontWeight: FontWeight.w600,
                                        ),
                                  ),
                                ],
                              ),
                            ),
                          ).animate().fadeIn(delay: 800.ms).slideX(begin: -0.2),
                          
                          const SizedBox(height: 16),
                          
                          // Create Profile Button - Glass style
                          Container(
                            width: double.infinity,
                            height: 56,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Colors.white.withOpacity(0.1),
                                  Colors.white.withOpacity(0.05),
                                ],
                              ),
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(
                                color: Colors.white.withOpacity(0.2),
                                width: 1.5,
                              ),
                            ),
                            child: ElevatedButton(
                              onPressed: () => context.go('/profile'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.transparent,
                                shadowColor: Colors.transparent,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16),
                                ),
                              ),
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Icon(Icons.person_add, color: Colors.white),
                                  const SizedBox(width: 12),
                                  Text(
                                    'Create Profile',
                                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                          color: Colors.white,
                                          fontWeight: FontWeight.w600,
                                        ),
                                  ),
                                ],
                              ),
                            ),
                          ).animate().fadeIn(delay: 1000.ms).slideX(begin: 0.2),
                        ],
                      ),
                    ),
                    
                    const SizedBox(height: 40),
                    
                    // Feature highlights
                    Wrap(
                      spacing: 20,
                      runSpacing: 20,
                      children: [
                        _buildFeatureChip('🎧 AI Analysis', context),
                        _buildFeatureChip('🚀 Real-time Upload', context),
                        _buildFeatureChip('🌟 Community', context),
                      ],
                    ).animate().fadeIn(delay: 1200.ms).slideY(begin: 0.3),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFeatureChip(String text, BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Colors.white.withOpacity(0.1),
            Colors.white.withOpacity(0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: Colors.white.withOpacity(0.1),
        ),
      ),
      child: Text(
        text,
        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: Colors.white.withOpacity(0.8),
              fontWeight: FontWeight.w500,
            ),
      ),
    );
  }
} 