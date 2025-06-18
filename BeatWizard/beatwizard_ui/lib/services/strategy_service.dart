import 'dart:convert';
import 'package:http/http.dart' as http;

class ActionCard {
  final String id;
  final String title;
  final String action;
  final String reason;
  final String caption;
  final List<String> hashtags;
  final int priority;

  ActionCard({
    required this.id,
    required this.title,
    required this.action,
    required this.reason,
    required this.caption,
    required this.hashtags,
    required this.priority,
  });

  factory ActionCard.fromJson(Map<String, dynamic> json) {
    return ActionCard(
      id: json['id'],
      title: json['title'],
      action: json['action'],
      reason: json['reason'],
      caption: json['caption'],
      hashtags: List<String>.from(json['hashtags']),
      priority: json['priority'],
    );
  }
}

class PlatformStats {
  final String platform;
  final int followers;
  final int totalPlays;
  final double engagementRate;
  final int recentContentCount;
  final List<Map<String, dynamic>> topContent;
  final Map<String, dynamic> metrics;

  PlatformStats({
    required this.platform,
    required this.followers,
    required this.totalPlays,
    required this.engagementRate,
    required this.recentContentCount,
    required this.topContent,
    required this.metrics,
  });

  factory PlatformStats.fromJson(Map<String, dynamic> json) {
    return PlatformStats(
      platform: json['platform'],
      followers: json['followers'],
      totalPlays: json['total_plays'],
      engagementRate: json['engagement_rate'].toDouble(),
      recentContentCount: json['recent_content_count'],
      topContent: List<Map<String, dynamic>>.from(json['top_content']),
      metrics: Map<String, dynamic>.from(json['metrics']),
    );
  }
}

class StrategyResponse {
  final int momentumScore;
  final Map<String, PlatformStats> platforms;
  final List<ActionCard> actionCards;
  final Map<String, dynamic> summary;

  StrategyResponse({
    required this.momentumScore,
    required this.platforms,
    required this.actionCards,
    required this.summary,
  });

  factory StrategyResponse.fromJson(Map<String, dynamic> json) {
    Map<String, PlatformStats> platforms = {};
    json['platforms'].forEach((key, value) {
      platforms[key] = PlatformStats.fromJson(value);
    });

    List<ActionCard> actionCards = [];
    for (var card in json['action_cards']) {
      actionCards.add(ActionCard.fromJson(card));
    }

    return StrategyResponse(
      momentumScore: json['momentum_score'],
      platforms: platforms,
      actionCards: actionCards,
      summary: Map<String, dynamic>.from(json['summary']),
    );
  }
}

class StrategyService {
  static const String baseUrl = 'http://localhost:8000';  // Change for production
  
  static Future<StrategyResponse> analyzePlatforms({
    String? spotify,
    String? tiktok,
    String? soundcloud,
    String? deezer,
  }) async {
    final Map<String, String> body = {};
    
    if (spotify != null && spotify.isNotEmpty) body['spotify'] = spotify;
    if (tiktok != null && tiktok.isNotEmpty) body['tiktok'] = tiktok;
    if (soundcloud != null && soundcloud.isNotEmpty) body['soundcloud'] = soundcloud;
    if (deezer != null && deezer.isNotEmpty) body['deezer'] = deezer;
    
    if (body.isEmpty) {
      throw Exception('At least one platform link is required');
    }
    
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/analyze'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return StrategyResponse.fromJson(data);
      } else {
        final error = jsonDecode(response.body);
        throw Exception(error['detail'] ?? 'Failed to analyze platforms');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }
} 