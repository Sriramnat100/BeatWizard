import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:math' as math;
import '../services/strategy_service.dart';
import '../services/ai_advisor_service.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class StrategyDashboard extends StatefulWidget {
  const StrategyDashboard({Key? key}) : super(key: key);

  @override
  _StrategyDashboardState createState() => _StrategyDashboardState();
}

class _StrategyDashboardState extends State<StrategyDashboard> {
  final _formKey = GlobalKey<FormState>();
  final _spotifyController = TextEditingController();
  final _tiktokController = TextEditingController();
  final _soundcloudController = TextEditingController();
  final _deezerController = TextEditingController();
  
  bool _isLoading = false;
  StrategyResponse? _response;
  String? _error;
  int _currentCardIndex = 0;
  Map<String, bool> _cardActions = {}; // Track accept/skip
  
  // AI Advisor state
  bool _isLoadingAI = false;
  AIAdviceResponse? _aiAdvice;
  String? _momentumInsights;

  @override
  void dispose() {
    _spotifyController.dispose();
    _tiktokController.dispose();
    _soundcloudController.dispose();
    _deezerController.dispose();
    super.dispose();
  }

  Future<void> _analyzeLinks() async {
    if (!_formKey.currentState!.validate()) return;
    
    setState(() {
      _isLoading = true;
      _error = null;
      _response = null;
      _currentCardIndex = 0;
      _cardActions.clear();
      _aiAdvice = null;
      _momentumInsights = null;
    });
    
    try {
      final response = await StrategyService.analyzePlatforms(
        spotify: _spotifyController.text,
        tiktok: _tiktokController.text,
        soundcloud: _soundcloudController.text,
        deezer: _deezerController.text,
      );
      
      setState(() {
        _response = response;
        _isLoading = false;
      });
      
      // Get AI insights after successful analysis
      _getAIInsights();
      
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<void> _getAIInsights() async {
    if (_response == null) return;
    
    setState(() {
      _isLoadingAI = true;
    });
    
    try {
      // Convert response to analytics format for AI
      final analyticsData = {
        'momentum_score': _response!.momentumScore,
        'platforms': _response!.platforms.map((key, value) => MapEntry(key, {
          'followers': value.followers,
          'total_plays': value.totalPlays,
          'engagement_rate': value.engagementRate,
          'recent_content_count': value.recentContentCount,
        })),
        'summary': _response!.summary,
      };
      
      // Get both personalized advice and momentum insights
      final futures = await Future.wait([
        AIAdvisorService.getPersonalizedAdvice(
          userAnalytics: analyticsData,
          adviceType: 'momentum',
        ),
        AIAdvisorService.getMomentumInsights(
          userAnalytics: analyticsData,
        ),
      ]);
      
      setState(() {
        _aiAdvice = futures[0] as AIAdviceResponse;
        _momentumInsights = futures[1] as String;
        _isLoadingAI = false;
      });
      
    } catch (e) {
      setState(() {
        _isLoadingAI = false;
      });
      print('AI insights error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'BeatWizard Strategy',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: _response == null
            ? _buildInputForm()
            : _buildResultsView(),
      ),
    );
  }

  Widget _buildInputForm() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Enter Your Platform Links',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Add at least one platform to get started',
              style: TextStyle(
                fontSize: 14,
                color: Colors.white60,
              ),
            ),
            const SizedBox(height: 24),
            
            _buildPlatformInput(
              controller: _spotifyController,
              label: 'Spotify Artist URL',
              icon: Icons.music_note,
              hint: 'https://open.spotify.com/artist/...',
              color: const Color(0xFF1DB954),
            ),
            
            _buildPlatformInput(
              controller: _tiktokController,
              label: 'TikTok Profile Link',
              icon: Icons.video_library,
              hint: 'https://www.tiktok.com/@username',
              color: const Color(0xFFFF0050),
            ),
            
            _buildPlatformInput(
              controller: _soundcloudController,
              label: 'SoundCloud Link',
              icon: Icons.cloud,
              hint: 'https://soundcloud.com/username',
              color: const Color(0xFFFF5500),
            ),
            
            _buildPlatformInput(
              controller: _deezerController,
              label: 'Deezer Artist URL',
              icon: Icons.album,
              hint: 'https://www.deezer.com/artist/...',
              color: const Color(0xFFFF0092),
            ),
            
            const SizedBox(height: 32),
            
            if (_error != null)
              Container(
                padding: const EdgeInsets.all(12),
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red.withOpacity(0.3)),
                ),
                child: Text(
                  _error!,
                  style: const TextStyle(color: Colors.red),
                ),
              ),
            
            SizedBox(
              width: double.infinity,
              height: 56,
              child: ElevatedButton(
                onPressed: _isLoading ? null : _analyzeLinks,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF6C63FF),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: _isLoading
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2,
                        ),
                      )
                    : const Text(
                        'Analyze My Platforms',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPlatformInput({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    required String hint,
    required Color color,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      child: TextFormField(
        controller: controller,
        style: const TextStyle(color: Colors.white),
        decoration: InputDecoration(
          labelText: label,
          labelStyle: TextStyle(color: color),
          hintText: hint,
          hintStyle: const TextStyle(color: Colors.white30),
          prefixIcon: Icon(icon, color: color),
          filled: true,
          fillColor: Colors.white.withOpacity(0.05),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide.none,
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: color, width: 2),
          ),
        ),
        validator: (value) {
          // At least one platform required
          if (_spotifyController.text.isEmpty &&
              _tiktokController.text.isEmpty &&
              _soundcloudController.text.isEmpty &&
              _deezerController.text.isEmpty) {
            return 'Please enter at least one platform link';
          }
          return null;
        },
      ),
    );
  }

  Widget _buildResultsView() {
    return SingleChildScrollView(
      child: Column(
        children: [
          _buildMomentumScore(),
          _buildPlatformStats(),
          _buildActionCards(),
          _buildBackButton(),
        ],
      ),
    );
  }

  Widget _buildMomentumScore() {
    final score = _response!.momentumScore;
    final color = score >= 70
        ? Colors.green
        : score >= 40
            ? Colors.orange
            : Colors.red;
    
    return Column(
      children: [
        // Original momentum score display
        Container(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              const Text(
                'Your Momentum Score',
                style: TextStyle(
                  fontSize: 18,
                  color: Colors.white70,
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: 150,
                height: 150,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      width: 150,
                      height: 150,
                      child: CircularProgressIndicator(
                        value: score / 100,
                        backgroundColor: Colors.white.withOpacity(0.1),
                        valueColor: AlwaysStoppedAnimation<Color>(color),
                        strokeWidth: 12,
                      ),
                    ),
                    Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          '$score',
                          style: TextStyle(
                            fontSize: 48,
                            fontWeight: FontWeight.bold,
                            color: color,
                          ),
                        ),
                        Text(
                          score >= 70
                              ? 'Excellent!'
                              : score >= 40
                                  ? 'Good'
                                  : 'Needs Work',
                          style: TextStyle(
                            fontSize: 16,
                            color: color,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        
        // AI Insights Section
        if (_isLoadingAI)
          Container(
            margin: const EdgeInsets.only(top: 20),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.purple.withOpacity(0.15),
                  Colors.purple.withOpacity(0.05),
                ],
              ),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.purple.withOpacity(0.2)),
            ),
            child: Column(
              children: [
                const CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.purple),
                ),
                const SizedBox(height: 16),
                Text(
                  '🤖 AI is analyzing your momentum...',
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.8),
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ).animate().fadeIn().slideY(begin: 0.3),
        
        // AI Insights Display
        if (_momentumInsights != null && !_isLoadingAI)
          Container(
            margin: const EdgeInsets.only(top: 20),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.purple.withOpacity(0.15),
                  Colors.purple.withOpacity(0.05),
                ],
              ),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.purple.withOpacity(0.2)),
              boxShadow: [
                BoxShadow(
                  color: Colors.purple.withOpacity(0.1),
                  blurRadius: 20,
                  spreadRadius: 5,
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.psychology,
                      color: Colors.purple.shade300,
                      size: 24,
                    ),
                    const SizedBox(width: 12),
                    Text(
                      'BeatWizard AI Insights',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.purple.shade300,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Text(
                  _momentumInsights!,
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white.withOpacity(0.9),
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ).animate().fadeIn(delay: 300.ms).slideY(begin: 0.3),
        
        // AI Personalized Advice
        if (_aiAdvice != null && !_isLoadingAI)
          Container(
            margin: const EdgeInsets.only(top: 20),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.blue.withOpacity(0.15),
                  Colors.blue.withOpacity(0.05),
                ],
              ),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.blue.withOpacity(0.2)),
              boxShadow: [
                BoxShadow(
                  color: Colors.blue.withOpacity(0.1),
                  blurRadius: 20,
                  spreadRadius: 5,
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.lightbulb_outline,
                      color: Colors.blue.shade300,
                      size: 24,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Personalized Growth Strategy',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.blue.shade300,
                        ),
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.blue.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        _aiAdvice!.priorityFocus,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.blue.shade200,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Text(
                  _aiAdvice!.advice,
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.white.withOpacity(0.9),
                    height: 1.5,
                  ),
                ),
                
                if (_aiAdvice!.actionItems.isNotEmpty) ...[
                  const SizedBox(height: 20),
                  Text(
                    'Action Items:',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.blue.shade300,
                    ),
                  ),
                  const SizedBox(height: 12),
                  ...(_aiAdvice!.actionItems.take(5).map((item) => Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          margin: const EdgeInsets.only(top: 6, right: 12),
                          width: 6,
                          height: 6,
                          decoration: BoxDecoration(
                            color: Colors.blue.shade300,
                            shape: BoxShape.circle,
                          ),
                        ),
                        Expanded(
                          child: Text(
                            item,
                            style: TextStyle(
                              fontSize: 15,
                              color: Colors.white.withOpacity(0.8),
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ))),
                ],
                
                const SizedBox(height: 16),
                Row(
                  children: [
                    Icon(
                      Icons.schedule,
                      color: Colors.blue.shade400,
                      size: 16,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Timeline: ${_aiAdvice!.estimatedTimeline}',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.blue.shade400,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(width: 20),
                    Icon(
                      Icons.verified,
                      color: Colors.green.shade400,
                      size: 16,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${(_aiAdvice!.confidenceScore * 100).round()}% Confidence',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.green.shade400,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ).animate().fadeIn(delay: 600.ms).slideY(begin: 0.3),
      ],
    );
  }

  Widget _buildPlatformStats() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Platform Performance',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 16),
          ..._response!.platforms.entries.map((entry) {
            final platform = entry.key;
            final stats = entry.value;
            return _buildPlatformCard(platform, stats);
          }).toList(),
        ],
      ),
    );
  }

  Widget _buildPlatformCard(String platform, PlatformStats stats) {
    final color = _getPlatformColor(platform);
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                platform.toUpperCase(),
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '${(stats.engagementRate * 100).toStringAsFixed(1)}% engagement',
                  style: TextStyle(
                    fontSize: 12,
                    color: color,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildStatItem('Followers', _formatNumber(stats.followers), color),
              _buildStatItem('Recent Posts', stats.recentContentCount.toString(), color),
            ],
          ),
          // Top Tracks Section (ONLY)
          if (stats.topContent.isNotEmpty) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withOpacity(0.05),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: color.withOpacity(0.2)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.trending_up,
                        color: color,
                        size: 16,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Top Tracks',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: color,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ...stats.topContent.take(3).map<Widget>((track) {
                    final title = track['title'] ?? 'Unknown Track';
                    final plays = track['plays'] ?? 0;
                    final truncatedTitle = title.length > 35 ? '${title.substring(0, 35)}...' : title;
                    final trackUrl = track['url'] ?? '';
                    return StatefulBuilder(
                      builder: (context, setState) {
                        bool loading = false;
                        Map<String, dynamic>? features;
                        return Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Container(
                              margin: const EdgeInsets.only(bottom: 6),
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                              decoration: BoxDecoration(
                                color: color.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(color: color.withOpacity(0.3)),
                              ),
                              child: Row(
                                children: [
                                  Container(
                                    width: 20,
                                    height: 20,
                                    decoration: BoxDecoration(
                                      color: color.withOpacity(0.3),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Icon(
                                      Icons.play_arrow,
                                      color: color,
                                      size: 12,
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          truncatedTitle,
                                          style: TextStyle(
                                            fontSize: 12,
                                            color: color,
                                            fontWeight: FontWeight.w600,
                                          ),
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis,
                                        ),
                                        if (plays > 0)
                                          Text(
                                            '${_formatNumber(plays)} plays',
                                            style: TextStyle(
                                              fontSize: 10,
                                              color: color.withOpacity(0.7),
                                            ),
                                          ),
                                      ],
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  ElevatedButton(
                                    onPressed: loading
                                        ? null
                                        : () async {
                                            setState(() => loading = true);
                                            try {
                                              final response = await http.post(
                                                Uri.parse('http://localhost:8000/extract_features'),
                                                headers: {'Content-Type': 'application/json'},
                                                body: jsonEncode({'track_url': trackUrl}),
                                              );
                                              if (response.statusCode == 200) {
                                                final data = jsonDecode(response.body);
                                                features = data['features'];
                                                showDialog(
                                                  context: context,
                                                  builder: (_) => AlertDialog(
                                                    title: Text('Raw Features'),
                                                    content: SingleChildScrollView(
                                                      child: Text(jsonEncode(features)),
                                                    ),
                                                    actions: [
                                                      TextButton(
                                                        onPressed: () => Navigator.of(context).pop(),
                                                        child: const Text('Close'),
                                                      ),
                                                    ],
                                                  ),
                                                );
                                              } else {
                                                showDialog(
                                                  context: context,
                                                  builder: (_) => AlertDialog(
                                                    title: Text('Error'),
                                                    content: Text('Failed to extract features.'),
                                                    actions: [
                                                      TextButton(
                                                        onPressed: () => Navigator.of(context).pop(),
                                                        child: const Text('Close'),
                                                      ),
                                                    ],
                                                  ),
                                                );
                                              }
                                            } catch (e) {
                                              showDialog(
                                                context: context,
                                                builder: (_) => AlertDialog(
                                                  title: Text('Error'),
                                                  content: Text('Network error: \$e'),
                                                  actions: [
                                                    TextButton(
                                                      onPressed: () => Navigator.of(context).pop(),
                                                      child: const Text('Close'),
                                                    ),
                                                  ],
                                                ),
                                              );
                                            } finally {
                                              setState(() => loading = false);
                                            }
                                          },
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: color,
                                      foregroundColor: Colors.white,
                                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                      textStyle: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold),
                                    ),
                                    child: loading
                                        ? const SizedBox(
                                            width: 14,
                                            height: 14,
                                            child: CircularProgressIndicator(
                                              color: Colors.white,
                                              strokeWidth: 2,
                                            ),
                                          )
                                        : const Text('Extract Features'),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        );
                      },
                    );
                  }).toList(),
                ],
              ),
            ),
          ],
          
          // Similar Artists Section (EXISTING)
          if (stats.metrics['similar_artists'] != null && 
              (stats.metrics['similar_artists'] as List).isNotEmpty) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withOpacity(0.05),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: color.withOpacity(0.2)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.people_outline,
                        color: color,
                        size: 16,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Fans Also Like',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: color,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 6,
                    children: (stats.metrics['similar_artists'] as List)
                        .take(5)
                        .map<Widget>((artist) => Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: color.withOpacity(0.3)),
                          ),
                          child: Text(
                            artist['name'],
                            style: TextStyle(
                              fontSize: 12,
                              color: color,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        )).toList(),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStatItem(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: Colors.white60,
          ),
        ),
      ],
    );
  }

  Widget _buildActionCards() {
    if (_response!.actionCards.isEmpty) {
      return const SizedBox.shrink();
    }
    
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Action Cards',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Swipe right to accept, left to skip',
            style: TextStyle(
              fontSize: 14,
              color: Colors.white60,
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 400,
            child: PageView.builder(
              controller: PageController(viewportFraction: 0.9),
              itemCount: _response!.actionCards.length,
              onPageChanged: (index) {
                setState(() {
                  _currentCardIndex = index;
                });
              },
              itemBuilder: (context, index) {
                final card = _response!.actionCards[index];
                return _buildSwipeableCard(card);
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSwipeableCard(ActionCard card) {
    return Dismissible(
      key: Key(card.id),
      onDismissed: (direction) {
        setState(() {
          _cardActions[card.id] = direction == DismissDirection.endToStart ? false : true;
        });
        
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              direction == DismissDirection.endToStart
                  ? 'Skipped: ${card.title}'
                  : 'Accepted: ${card.title}',
            ),
            backgroundColor: direction == DismissDirection.endToStart
                ? Colors.orange
                : Colors.green,
            duration: const Duration(seconds: 2),
          ),
        );
      },
      child: GestureDetector(
        onTap: () => _showCardDetails(card),
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 8),
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                const Color(0xFF6C63FF),
                const Color(0xFF6C63FF).withOpacity(0.7),
              ],
            ),
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF6C63FF).withOpacity(0.3),
                blurRadius: 20,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(
                      Icons.lightbulb_outline,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      card.title,
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Text(
                card.action,
                style: const TextStyle(
                  fontSize: 16,
                  color: Colors.white,
                  height: 1.5,
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.touch_app,
                      color: Colors.white,
                      size: 16,
                    ),
                    SizedBox(width: 4),
                    Text(
                      'Tap for details',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showCardDetails(ActionCard card) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => Container(
        decoration: const BoxDecoration(
          color: Color(0xFF1A1A2E),
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              card.title,
              style: const TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.orange.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.orange.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: const [
                      Icon(Icons.info_outline, color: Colors.orange, size: 20),
                      SizedBox(width: 8),
                      Text(
                        'Why this matters',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.orange,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    card.reason,
                    style: const TextStyle(
                      fontSize: 14,
                      color: Colors.white70,
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'Suggested Caption',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.05),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white.withOpacity(0.1)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    card.caption,
                    style: const TextStyle(
                      fontSize: 14,
                      color: Colors.white,
                      height: 1.5,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    children: card.hashtags
                        .map((tag) => Chip(
                              label: Text(
                                tag,
                                style: const TextStyle(
                                  color: Color(0xFF6C63FF),
                                  fontSize: 12,
                                ),
                              ),
                              backgroundColor: const Color(0xFF6C63FF).withOpacity(0.1),
                              side: BorderSide(
                                color: const Color(0xFF6C63FF).withOpacity(0.3),
                              ),
                            ))
                        .toList(),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              height: 50,
              child: ElevatedButton.icon(
                onPressed: () {
                  Clipboard.setData(ClipboardData(
                    text: '${card.caption}\n\n${card.hashtags.join(' ')}',
                  ));
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Caption copied to clipboard!'),
                      backgroundColor: Colors.green,
                    ),
                  );
                },
                icon: const Icon(Icons.copy, color: Colors.white),
                label: const Text(
                  'Copy Caption',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF6C63FF),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBackButton() {
    return Container(
      padding: const EdgeInsets.all(20),
      child: TextButton(
        onPressed: () {
          setState(() {
            _response = null;
            _error = null;
          });
        },
        child: const Text(
          'Analyze Different Platforms',
          style: TextStyle(
            color: Color(0xFF6C63FF),
            fontSize: 16,
          ),
        ),
      ),
    );
  }

  Color _getPlatformColor(String platform) {
    switch (platform.toLowerCase()) {
      case 'spotify':
        return const Color(0xFF1DB954);
      case 'tiktok':
        return const Color(0xFFFF0050);
      case 'soundcloud':
        return const Color(0xFFFF5500);
      case 'deezer':
        return const Color(0xFFFF0092);
      default:
        return const Color(0xFF6C63FF);
    }
  }

  String _formatNumber(int number) {
    if (number >= 1000000) {
      return '${(number / 1000000).toStringAsFixed(1)}M';
    } else if (number >= 1000) {
      return '${(number / 1000).toStringAsFixed(1)}K';
    }
    return number.toString();
  }
} 