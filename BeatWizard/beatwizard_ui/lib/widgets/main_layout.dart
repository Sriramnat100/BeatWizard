import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:flutter_animate/flutter_animate.dart';

class MainLayout extends StatefulWidget {
  final Widget child;
  final String currentRoute;

  const MainLayout({
    super.key,
    required this.child,
    required this.currentRoute,
  });

  @override
  State<MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> with TickerProviderStateMixin {
  late TabController _tabController;
  late AnimationController _backgroundController;

  int _getSelectedIndex() {
    switch (widget.currentRoute) {
      case '/strategy':
        return 0;
      case '/explore':
        return 1;
      case '/search':
        return 2;
      case '/upload':
        return 3;
      case '/profile':
      case '/profile/display':
        return 4;
      default:
        return 0; // Default to strategy
    }
  }

  @override
  void initState() {
    super.initState();
    _tabController = TabController(
      length: 5,
      vsync: this,
      initialIndex: _getSelectedIndex(),
    );
    
    _backgroundController = AnimationController(
      duration: const Duration(seconds: 20),
      vsync: this,
    )..repeat();
  }

  @override
  void didUpdateWidget(MainLayout oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.currentRoute != widget.currentRoute) {
      _tabController.animateTo(_getSelectedIndex());
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    _backgroundController.dispose();
    super.dispose();
  }

  void _onTabTapped(int index) {
    switch (index) {
      case 0:
        context.go('/strategy');
        break;
      case 1:
        context.go('/explore');
        break;
      case 2:
        context.go('/search');
        break;
      case 3:
        context.go('/upload');
        break;
      case 4:
        context.go('/profile/display'); // Always go to profile display
        break;
    }
  }

  Widget _buildBeatWizardLogo() {
    return Container(
      width: 32,
      height: 32,
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
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
            blurRadius: 8,
            spreadRadius: 2,
          ),
        ],
      ),
      child: Stack(
        children: [
          // Wizard hat
          Positioned(
            top: 2,
            left: 8,
            right: 8,
            child: Container(
              height: 14,
              decoration: const BoxDecoration(
                color: Color(0xFF8B4D8E),
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(12),
                  topRight: Radius.circular(12),
                ),
              ),
            ),
          ),
          // Stars
          const Positioned(
            top: 6,
            left: 12,
            child: Icon(Icons.star, color: Color(0xFFF59E0B), size: 6),
          ),
          const Positioned(
            top: 4,
            right: 10,
            child: Icon(Icons.star, color: Color(0xFFF59E0B), size: 4),
          ),
          // Turntables
          Positioned(
            bottom: 2,
            left: 4,
            child: Container(
              width: 8,
              height: 8,
              decoration: const BoxDecoration(
                color: Color(0xFF374151),
                shape: BoxShape.circle,
              ),
              child: Center(
                child: Container(
                  width: 4,
                  height: 4,
                  decoration: const BoxDecoration(
                    color: Color(0xFFF59E0B),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            bottom: 2,
            right: 4,
            child: Container(
              width: 8,
              height: 8,
              decoration: const BoxDecoration(
                color: Color(0xFF374151),
                shape: BoxShape.circle,
              ),
              child: Center(
                child: Container(
                  width: 4,
                  height: 4,
                  decoration: const BoxDecoration(
                    color: Color(0xFFF59E0B),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
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
          ),
        ),
        child: Stack(
          children: [
            // Animated background elements
            AnimatedBuilder(
              animation: _backgroundController,
              builder: (context, child) {
                return Stack(
                  children: List.generate(15, (index) {
                    final offset = _backgroundController.value * 2 * 3.14159;
                    return Positioned(
                      left: 50.0 + (index * 80.0) % MediaQuery.of(context).size.width,
                      top: 50.0 + (index * 120.0 + offset * 20) % MediaQuery.of(context).size.height,
                      child: Container(
                        width: 3,
                        height: 3,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.05),
                          shape: BoxShape.circle,
                        ),
                      ),
                    );
                  }),
                );
              },
            ),
            
            // Main content area
            Column(
              children: [
                // Custom App Bar
                Container(
                  height: 120,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Colors.black.withOpacity(0.3),
                        Colors.transparent,
                      ],
                    ),
                  ),
                  child: SafeArea(
                    child: Container(
                      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
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
                          width: 1,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.1),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                      child: Row(
                        children: [
                          // Logo section
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Theme.of(context).colorScheme.primary.withOpacity(0.3),
                                  Theme.of(context).colorScheme.primary.withOpacity(0.1),
                                ],
                              ),
                              borderRadius: BorderRadius.circular(12),
                              boxShadow: [
                                BoxShadow(
                                  color: Theme.of(context).colorScheme.primary.withOpacity(0.2),
                                  blurRadius: 10,
                                  spreadRadius: 2,
                                ),
                              ],
                            ),
                            child: _buildBeatWizardLogo(),
                          ),
                          const SizedBox(width: 12),
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
                              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w700,
                                    fontSize: 20,
                                  ),
                            ),
                          ),
                          const Spacer(),
                          // Status indicator
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Colors.green.withOpacity(0.2),
                                  Colors.green.withOpacity(0.1),
                                ],
                              ),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: Colors.green.withOpacity(0.3)),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Container(
                                  width: 6,
                                  height: 6,
                                  decoration: const BoxDecoration(
                                    color: Colors.green,
                                    shape: BoxShape.circle,
                                  ),
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  'Online',
                                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                        color: Colors.green,
                                        fontWeight: FontWeight.w500,
                                      ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ).animate().fadeIn().slideY(begin: -0.5),
                
                // Content area
                Expanded(
                  child: widget.child,
                ),
                
                // Bottom Navigation Bar
                Container(
                  margin: const EdgeInsets.all(20),
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Colors.black.withOpacity(0.3),
                        Colors.black.withOpacity(0.1),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(25),
                    border: Border.all(
                      color: Colors.white.withOpacity(0.1),
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.2),
                        blurRadius: 20,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(25),
                    child: TabBar(
                      controller: _tabController,
                      onTap: _onTabTapped,
                      indicator: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Theme.of(context).colorScheme.primary.withOpacity(0.3),
                            Theme.of(context).colorScheme.primary.withOpacity(0.1),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      labelColor: Colors.white,
                      unselectedLabelColor: Colors.white.withOpacity(0.5),
                      indicatorSize: TabBarIndicatorSize.tab,
                      dividerColor: Colors.transparent,
                      splashBorderRadius: BorderRadius.circular(20),
                      tabs: [
                        _buildNavTab(
                          icon: Icons.analytics_outlined,
                          selectedIcon: Icons.analytics,
                          label: 'Strategy',
                          isSelected: _getSelectedIndex() == 0,
                        ),
                        _buildNavTab(
                          icon: Icons.explore_outlined,
                          selectedIcon: Icons.explore,
                          label: 'Explore',
                          isSelected: _getSelectedIndex() == 1,
                        ),
                        _buildNavTab(
                          icon: Icons.search_outlined,
                          selectedIcon: Icons.search,
                          label: 'Search',
                          isSelected: _getSelectedIndex() == 2,
                        ),
                        _buildNavTab(
                          icon: Icons.add_circle_outline,
                          selectedIcon: Icons.add_circle,
                          label: 'Upload',
                          isSelected: _getSelectedIndex() == 3,
                        ),
                        _buildNavTab(
                          icon: Icons.person_outline,
                          selectedIcon: Icons.person,
                          label: 'Profile',
                          isSelected: _getSelectedIndex() == 4,
                        ),
                      ],
                    ),
                  ),
                ).animate().fadeIn(delay: 300.ms).slideY(begin: 0.5),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNavTab({
    required IconData icon,
    required IconData selectedIcon,
    required String label,
    required bool isSelected,
  }) {
    return Container(
      height: 60,
      child: Tab(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: isSelected
                  ? BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Theme.of(context).colorScheme.primary.withOpacity(0.3),
                          Theme.of(context).colorScheme.primary.withOpacity(0.1),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(12),
                      boxShadow: [
                        BoxShadow(
                          color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                          blurRadius: 8,
                          spreadRadius: 2,
                        ),
                      ],
                    )
                  : null,
              child: Icon(
                isSelected ? selectedIcon : icon,
                size: 24,
                color: isSelected 
                    ? Theme.of(context).colorScheme.primary 
                    : Colors.white.withOpacity(0.7),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: isSelected 
                        ? Colors.white 
                        : Colors.white.withOpacity(0.5),
                    fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
                    fontSize: 11,
                  ),
            ),
          ],
        ),
      ),
    );
  }
} 