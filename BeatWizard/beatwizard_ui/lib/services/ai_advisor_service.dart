import 'dart:convert';
import 'package:http/http.dart' as http;

class AIAdvisorService {
  static const String baseUrl = 'http://localhost:8001';

  static Future<AIAdviceResponse> getPersonalizedAdvice({
    required Map<String, dynamic> userAnalytics,
    String adviceType = 'momentum',
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/advice'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_analytics': userAnalytics,
          'advice_type': adviceType,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return AIAdviceResponse.fromJson(data);
      } else {
        final error = jsonDecode(response.body);
        throw Exception(error['detail'] ?? 'Failed to get AI advice');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  static Future<String> getMomentumInsights({
    required Map<String, dynamic> userAnalytics,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/momentum-insights'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_analytics': userAnalytics,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['insights'] as String;
      } else {
        final error = jsonDecode(response.body);
        throw Exception(error['detail'] ?? 'Failed to get momentum insights');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  static Future<AIChatResponse> chatWithAI({
    required String message,
    Map<String, dynamic>? userAnalytics,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/chat'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'message': message,
          'user_analytics': userAnalytics,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return AIChatResponse.fromJson(data);
      } else {
        final error = jsonDecode(response.body);
        throw Exception(error['detail'] ?? 'Failed to chat with AI');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }
}

class AIAdviceResponse {
  final String advice;
  final List<String> actionItems;
  final String priorityFocus;
  final String estimatedTimeline;
  final double confidenceScore;

  AIAdviceResponse({
    required this.advice,
    required this.actionItems,
    required this.priorityFocus,
    required this.estimatedTimeline,
    required this.confidenceScore,
  });

  factory AIAdviceResponse.fromJson(Map<String, dynamic> json) {
    return AIAdviceResponse(
      advice: json['advice'] as String,
      actionItems: List<String>.from(json['action_items'] ?? []),
      priorityFocus: json['priority_focus'] as String,
      estimatedTimeline: json['estimated_timeline'] as String,
      confidenceScore: (json['confidence_score'] as num).toDouble(),
    );
  }
}

class AIChatResponse {
  final String response;
  final List<String> suggestions;
  final String timestamp;

  AIChatResponse({
    required this.response,
    required this.suggestions,
    required this.timestamp,
  });

  factory AIChatResponse.fromJson(Map<String, dynamic> json) {
    return AIChatResponse(
      response: json['response'] as String,
      suggestions: List<String>.from(json['suggestions'] ?? []),
      timestamp: json['timestamp'] as String,
    );
  }
} 