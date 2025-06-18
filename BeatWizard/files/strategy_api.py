from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from link_based_analytics import LinkBasedAnalyticsExtractor
from datetime import datetime
import json
import os
import yt_dlp
from beat_analyzer import AudioFeatureExtractor

app = FastAPI()

# Enable CORS for Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlatformLinksRequest(BaseModel):
    spotify: Optional[str] = ""
    tiktok: Optional[str] = ""
    soundcloud: Optional[str] = ""
    deezer: Optional[str] = ""

class ActionCard(BaseModel):
    id: str
    title: str
    action: str
    reason: str
    caption: str
    hashtags: List[str]
    priority: int  # 1-3, higher is more important

class PlatformStats(BaseModel):
    platform: str
    followers: int
    total_plays: int
    engagement_rate: float
    recent_content_count: int
    top_content: List[Dict]
    metrics: Dict

class StrategyResponse(BaseModel):
    momentum_score: int
    platforms: Dict[str, PlatformStats]
    action_cards: List[ActionCard]
    summary: Dict

@app.post("/analyze", response_model=StrategyResponse)
async def analyze_platforms(request: PlatformLinksRequest):
    """Analyze platform links and return strategy recommendations"""
    
    # Collect non-empty URLs
    urls = []
    url_map = {}
    
    if request.spotify:
        urls.append(request.spotify)
        url_map[request.spotify] = 'spotify'
    if request.tiktok:
        urls.append(request.tiktok)
        url_map[request.tiktok] = 'tiktok'
    if request.soundcloud:
        urls.append(request.soundcloud)
        url_map[request.soundcloud] = 'soundcloud'
    if request.deezer:
        urls.append(request.deezer)
        url_map[request.deezer] = 'deezer'
    
    if not urls:
        raise HTTPException(status_code=400, detail="Please provide at least one platform link")
    
    # Extract analytics
    extractor = LinkBasedAnalyticsExtractor()
    platform_data = {}
    
    try:
        for url in urls:
            result = extractor.extract_from_url(url)
            if result:
                platform = result.platform.lower()
                platform_data[platform] = result
    finally:
        extractor.cleanup()
    
    if not platform_data:
        raise HTTPException(status_code=400, detail="Could not extract data from any platform")
    
    # Calculate momentum score
    momentum_score = calculate_momentum_score(platform_data)
    
    # Generate action cards
    action_cards = generate_smart_action_cards(platform_data)
    
    # Format response
    platforms = {}
    for platform, data in platform_data.items():
        platforms[platform] = PlatformStats(
            platform=platform,
            followers=data.followers,
            total_plays=data.total_plays,
            engagement_rate=data.engagement_rate,
            recent_content_count=len(data.recent_posts),
            top_content=[
                {
                    'title': content.get('title', 'Unknown'),
                    'plays': content.get('plays', content.get('views', 0)),
                    'engagement': content.get('engagement', 0)
                }
                for content in data.top_content[:3]
            ],
            metrics=data.growth_metrics
        )
    
    # Create summary
    summary = {
        'total_reach': sum(p.followers for p in platform_data.values()),
        'total_plays': sum(p.total_plays for p in platform_data.values()),
        'average_engagement': sum(p.engagement_rate for p in platform_data.values()) / len(platform_data),
        'platform_count': len(platform_data),
        'strongest_platform': max(platform_data.keys(), key=lambda k: platform_data[k].followers)
    }
    
    return StrategyResponse(
        momentum_score=momentum_score,
        platforms=platforms,
        action_cards=action_cards,
        summary=summary
    )

def calculate_momentum_score(platform_data: Dict) -> int:
    """Calculate momentum score with weighted factors"""
    score = 0
    weights = {
        'followers': 0.25,
        'engagement': 0.35,
        'content': 0.20,
        'diversity': 0.20
    }
    
    # Platform diversity bonus
    diversity_score = min(len(platform_data) * 20, 60)  # Max 60 points for 3+ platforms
    
    total_weight = 0
    
    for platform, data in platform_data.items():
        platform_score = 0
        
        # Follower score (logarithmic scale)
        if data.followers > 0:
            import math
            follower_score = min(math.log10(data.followers + 1) * 15, 40)
            platform_score += follower_score * weights['followers']
        
        # Engagement score
        if data.engagement_rate > 0:
            # High engagement is valuable
            if data.engagement_rate > 0.10:  # 10%+ is excellent
                engagement_score = 100
            elif data.engagement_rate > 0.05:  # 5%+ is very good
                engagement_score = 80
            elif data.engagement_rate > 0.02:  # 2%+ is good
                engagement_score = 60
            else:
                engagement_score = data.engagement_rate * 2000  # Linear below 2%
            
            platform_score += engagement_score * weights['engagement']
        
        # Content consistency score
        content_score = min(len(data.recent_posts) * 10, 80)
        platform_score += content_score * weights['content']
        
        score += platform_score
        total_weight += 1
    
    # Average across platforms and add diversity bonus
    if total_weight > 0:
        score = (score / total_weight) + (diversity_score * weights['diversity'])
    
    return min(max(int(score), 0), 100)

def generate_smart_action_cards(platform_data: Dict) -> List[ActionCard]:
    """Generate data-driven action cards with specific insights"""
    cards = []
    card_id = 0
    
    # Analyze platform performance
    platform_scores = {}
    for platform, data in platform_data.items():
        # Calculate a composite score for each platform
        score = (data.followers * 0.3) + (data.engagement_rate * 10000 * 0.4) + (len(data.recent_posts) * 0.3)
        platform_scores[platform] = {
            'score': score,
            'data': data
        }
    
    # Sort platforms by performance
    sorted_platforms = sorted(platform_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Card 1: Cross-platform leverage opportunity
    if len(sorted_platforms) >= 2:
        strongest = sorted_platforms[0]
        weakest = sorted_platforms[-1]
        
        if strongest[1]['score'] > weakest[1]['score'] * 1.5:
            strong_name = strongest[0].title()
            weak_name = weakest[0].title()
            strong_followers = strongest[1]['data'].followers
            weak_followers = weakest[1]['data'].followers
            
            cards.append(ActionCard(
                id=f"card_{card_id}",
                title=f"Boost {weak_name} with {strong_name} Power",
                action=f"Create exclusive {weak_name} content and tease it on {strong_name}",
                reason=f"Your {strong_name} has {strong_followers:,} followers but {weak_name} only has {weak_followers:,}. Cross-promotion could grow {weak_name} by 20-50% in 30 days.",
                caption=f"🎵 Dropping something special on {weak_name} tomorrow! 🔥 First 100 fans get early access. Link in bio 👆 Who's ready? 🚀",
                hashtags=[f"#{strong_name}To{weak_name}", "#NewMusic", "#ExclusiveDrop"],
                priority=3
            ))
            card_id += 1
    
    # Card 2: Engagement optimization
    low_engagement_platforms = []
    for platform, data in platform_data.items():
        if data.engagement_rate < 0.03 and data.followers > 50:  # Less than 3% engagement
            low_engagement_platforms.append((platform, data))
    
    if low_engagement_platforms:
        platform, data = low_engagement_platforms[0]
        current_engagement = data.engagement_rate * 100
        
        cards.append(ActionCard(
            id=f"card_{card_id}",
            title=f"Fix {platform.title()} Engagement Drop",
            action="Post a poll or behind-the-scenes content today",
            reason=f"Your {platform.title()} engagement is only {current_engagement:.1f}%. Interactive content typically gets 3-5x more engagement. This could add 500+ interactions.",
            caption="🎹 Studio dilemma! Which vibe for the next track? 🤔\n\nA) Dark trap vibes 🌙\nB) Melodic summer anthem ☀️\nC) Experimental fusion 🎪\n\nComment below! Making this FOR YOU 💜",
            hashtags=["#StudioLife", "#ProducerCommunity", "#YourChoice"],
            priority=3
        ))
        card_id += 1
    
    # Card 3: Content consistency
    inactive_platforms = []
    for platform, data in platform_data.items():
        if len(data.recent_posts) < 3:
            inactive_platforms.append((platform, data))
    
    if inactive_platforms:
        platform, data = inactive_platforms[0]
        post_count = len(data.recent_posts)
        
        cards.append(ActionCard(
            id=f"card_{card_id}",
            title=f"Reactivate {platform.title()} Growth",
            action="Start a weekly content series",
            reason=f"You only have {post_count} recent posts on {platform.title()}. Regular posting can increase reach by 50%+ and add 100-500 followers/month.",
            caption="🎵 Introducing #MelodyMonday! 🎹\n\nEvery Monday = New beat snippet\nEvery Wednesday = Production tip\nEvery Friday = Full track preview\n\nTurn on notifications so you don't miss out! 🔔",
            hashtags=["#MelodyMonday", "#ContentSeries", "#WeeklyBeats"],
            priority=2
        ))
        card_id += 1
    
    # Card 4: Platform-specific opportunities
    if 'tiktok' in platform_data and platform_data['tiktok'].followers > 500:
        cards.append(ActionCard(
            id=f"card_{card_id}",
            title="Launch TikTok Sound Campaign",
            action="Upload your hook as an original sound",
            reason=f"With {platform_data['tiktok'].followers:,} TikTok followers, your sound could reach 10-100x your audience through user-generated content.",
            caption="🎵 NEW SOUND ALERT! Use this beat in your video and I'll duet my favorites! 🔥 Tag me @yourusername #ProducerSound",
            hashtags=["#OriginalSound", "#ProducerChallenge", "#UseMySound"],
            priority=2
        ))
        card_id += 1
    
    # Card 5: Growth momentum
    total_followers = sum(p.followers for p in platform_data.values())
    if total_followers > 1000:
        cards.append(ActionCard(
            id=f"card_{card_id}",
            title="Monetization Opportunity",
            action="Set up a beat store or offer custom production",
            reason=f"With {total_followers:,} total followers, you're ready to monetize. Producers with similar reach earn $200-1000+/month.",
            caption="🔥 BEAT STORE NOW LIVE! 💰\n\nExclusive beats for serious artists\n✅ Tagged: $25\n✅ Untagged: $50\n✅ Exclusive rights: $150\n\nDM for custom production! 🎹",
            hashtags=["#BeatStore", "#CustomBeats", "#ProducerLife"],
            priority=1
        ))
        card_id += 1
    
    # Card 6: Similar Artists Collaboration (NEW)
    for platform, data in platform_data.items():
        similar_artists = data.growth_metrics.get('similar_artists', [])
        if similar_artists and len(similar_artists) >= 2:
            top_similar = similar_artists[:3]  # Get top 3 similar artists
            artist_names = [artist['name'] for artist in top_similar]
            
            cards.append(ActionCard(
                id=f"card_{card_id}",
                title="Connect with Your Musical DNA",
                action=f"Reach out to {artist_names[0]} and {artist_names[1]} for potential collaboration",
                reason=f"Fans also like: {', '.join(artist_names)}. Collaborating with similar artists can increase your reach by 30-200% and bring authentic cross-pollination of audiences.",
                caption=f"🎵 Shoutout to {artist_names[0]} for the inspiration! Your latest track had me in the studio all night 🔥\n\nWho else is creating similar vibes? Drop your SoundCloud below 👇\n\n#CollabReady #ProducerNetwork",
                hashtags=["#CollabReady", "#ProducerNetwork", "#SimilarVibes"],
                priority=2
            ))
            card_id += 1
            break  # Only create one similar artists card
    
    # Sort by priority and limit to 6
    cards.sort(key=lambda x: x.priority, reverse=True)
    return cards[:6]  # Limit to 6 action cards

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/extract_features")
async def extract_features(request: Request):
    data = await request.json()
    track_url = data.get("track_url")
    if not track_url:
        return JSONResponse({"error": "No track_url provided"}, status_code=400)
    try:
        # Download the track
        output_dir = "downloads"
        os.makedirs(output_dir, exist_ok=True)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(track_url, download=True)
            audio_path = ydl.prepare_filename(info)
        # Extract features
        extractor = AudioFeatureExtractor()
        features = extractor.extract_features(audio_path)
        # Clean up
        try:
            os.remove(audio_path)
        except Exception:
            pass
        return {"features": features}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 