#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import openai
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="BeatWizard AI Advisor")

# Enable CORS for Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI
openai.api_key = "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"

class UserAnalytics(BaseModel):
    momentum_score: int
    platforms: Dict[str, Any]
    summary: Dict[str, Any]
    
class ChatRequest(BaseModel):
    message: str
    user_analytics: Optional[UserAnalytics] = None
    
class AdviceRequest(BaseModel):
    user_analytics: UserAnalytics
    advice_type: str = "general"  # general, growth, content, engagement

class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]
    timestamp: str

class AdviceResponse(BaseModel):
    advice: str
    action_items: List[str]
    priority_focus: str
    estimated_timeline: str
    confidence_score: float

def create_system_prompt(analytics_data: Optional[Dict] = None) -> str:
    """Create a specialized system prompt for music industry advice"""
    
    base_prompt = """You are BeatWizard AI, an expert music industry advisor specializing in helping beat makers, producers, and artists grow their audience and career. 

Your expertise includes:
- Social media growth strategies (SoundCloud, TikTok, Instagram, Spotify)
- Music marketing and promotion
- Beat making and production trends
- Platform-specific content strategies
- Audience engagement techniques
- Music industry networking
- Monetization strategies for producers

Always provide:
1. Specific, actionable advice
2. Realistic timelines
3. Platform-specific strategies
4. Current industry trends
5. Measurable goals

Keep responses concise but valuable. Use music industry terminology appropriately."""

    if analytics_data:
        analytics_context = f"""

CURRENT USER DATA:
- Momentum Score: {analytics_data.get('momentum_score', 'Unknown')}
- Total Reach: {analytics_data.get('summary', {}).get('total_reach', 0):,} followers
- Platform Count: {analytics_data.get('summary', {}).get('platform_count', 0)}
- Strongest Platform: {analytics_data.get('summary', {}).get('strongest_platform', 'None')}

PLATFORM BREAKDOWN:"""
        
        for platform, data in analytics_data.get('platforms', {}).items():
            analytics_context += f"""
- {platform.title()}: {data.get('followers', 0):,} followers, {data.get('recent_content_count', 0)} recent posts, {data.get('engagement_rate', 0):.1%} engagement"""
        
        return base_prompt + analytics_context
    
    return base_prompt

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """General chat endpoint for BeatWizard AI"""
    try:
        analytics_dict = request.user_analytics.dict() if request.user_analytics else None
        system_prompt = create_system_prompt(analytics_dict)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        # Extract suggestions (look for bullet points or numbered lists)
        suggestions = []
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith('•') or line.startswith('-') or 
                line.startswith('1.') or line.startswith('2.') or 
                line.startswith('3.') or line.startswith('4.') or 
                line.startswith('5.')):
                suggestions.append(line.lstrip('•-123456789. '))
        
        return ChatResponse(
            response=ai_response,
            suggestions=suggestions[:5],  # Limit to 5 suggestions
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")

@app.post("/advice", response_model=AdviceResponse)
async def get_personalized_advice(request: AdviceRequest):
    """Get personalized advice based on user analytics and advice type"""
    try:
        analytics_dict = request.user_analytics.dict()
        system_prompt = create_system_prompt(analytics_dict)
        
        # Create specialized prompts based on advice type
        advice_prompts = {
            "general": "Based on my current analytics, what should be my top 3 priorities to grow my music career?",
            "growth": "How can I increase my follower count and reach across platforms? Give me a specific 30-day action plan.",
            "content": "What type of content should I create to maximize engagement? Include posting frequency and content ideas.",
            "engagement": "My engagement rate is low. What specific strategies can I use to improve fan interaction?",
            "momentum": "Based on my momentum score and platform performance, what's my biggest opportunity for growth right now?"
        }
        
        user_prompt = advice_prompts.get(request.advice_type, advice_prompts["general"])
        
        # Add specific context based on analytics
        momentum_score = analytics_dict.get('momentum_score', 0)
        total_reach = analytics_dict.get('summary', {}).get('total_reach', 0)
        platform_count = analytics_dict.get('summary', {}).get('platform_count', 0)
        
        context_addition = f"""

My current situation:
- Momentum score: {momentum_score}/100
- Total reach: {total_reach:,} followers
- Active on {platform_count} platforms
- Need advice that's realistic for my current level

Please provide specific, actionable advice with clear next steps."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + context_addition}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        # Extract action items
        action_items = []
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith('•') or line.startswith('-') or 
                line.startswith('1.') or line.startswith('2.') or 
                line.startswith('3.') or line.startswith('4.') or 
                line.startswith('5.')):
                action_items.append(line.lstrip('•-123456789. '))
        
        # Determine priority focus based on momentum score
        if momentum_score < 20:
            priority_focus = "Foundation Building"
            timeline = "2-3 months"
            confidence = 0.9
        elif momentum_score < 50:
            priority_focus = "Consistent Growth"
            timeline = "1-2 months"
            confidence = 0.8
        elif momentum_score < 75:
            priority_focus = "Optimization & Scaling"
            timeline = "3-4 weeks"
            confidence = 0.7
        else:
            priority_focus = "Advanced Strategies"
            timeline = "2-3 weeks"
            confidence = 0.9
        
        return AdviceResponse(
            advice=ai_response,
            action_items=action_items[:6],  # Limit to 6 action items
            priority_focus=priority_focus,
            estimated_timeline=timeline,
            confidence_score=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advice generation failed: {str(e)}")

@app.post("/momentum-insights")
async def get_momentum_insights(request: AdviceRequest):
    """Get specific insights about momentum score and how to improve it"""
    try:
        analytics_dict = request.user_analytics.dict()
        momentum_score = analytics_dict.get('momentum_score', 0)
        
        system_prompt = f"""You are a music industry momentum expert. Analyze this artist's momentum score of {momentum_score}/100 and provide specific insights.

Focus on:
1. What this momentum score means
2. Specific actions to improve it
3. Realistic timeline for improvement
4. Platform-specific recommendations

Current data: {json.dumps(analytics_dict, indent=2)}

Be specific and actionable."""

        user_prompt = f"My momentum score is {momentum_score}/100. Explain what this means and give me a specific plan to improve it."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=400,
            temperature=0.6
        )
        
        return {"insights": response.choices[0].message.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Momentum insights failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BeatWizard AI Advisor", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    print("🤖 Starting BeatWizard AI Advisor on http://localhost:8001")
    print("🎵 Ready to provide personalized music industry advice!")
    uvicorn.run(app, host="0.0.0.0", port=8001) 