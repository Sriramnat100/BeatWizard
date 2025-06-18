import requests
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from urllib.parse import quote, urljoin, urlparse
import re
from bs4 import BeautifulSoup
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

@dataclass
class PlatformAnalytics:
    platform: str
    username: str
    profile_url: str
    followers: int
    total_plays: int
    recent_posts: List[Dict]
    engagement_rate: float
    top_content: List[Dict]
    growth_metrics: Dict
    last_updated: str

class LinkBasedAnalyticsExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.driver = None
        
        # User agents for requests
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15',
        ]
        
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
    def _setup_selenium_driver(self):
        """Setup Chrome driver with stealth options"""
        if self.driver:
            return self.driver
            
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={random.choice(self.user_agents)}')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            return self.driver
        except Exception as e:
            print(f"❌ Could not setup Selenium driver: {e}")
            return None
    
    def _random_delay(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """Add random delay to avoid detection"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
        
    def _get_page_with_selenium(self, url: str, wait_seconds: int = 5) -> Optional[BeautifulSoup]:
        """Get page content using Selenium"""
        driver = self._setup_selenium_driver()
        if not driver:
            return None
            
        try:
            print(f"🔍 Loading {url}...")
            driver.get(url)
            time.sleep(wait_seconds)  # Wait for dynamic content
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            return soup
            
        except Exception as e:
            print(f"❌ Error loading page: {e}")
            return None
    
    def _detect_platform(self, url: str) -> str:
        """Detect platform from URL"""
        url_lower = url.lower()
        
        if 'soundcloud.com' in url_lower:
            return 'soundcloud'
        elif 'tiktok.com' in url_lower:
            return 'tiktok'
        elif 'instagram.com' in url_lower:
            return 'instagram'
        elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            return 'youtube'
        elif 'spotify.com' in url_lower:
            return 'spotify'
        elif 'deezer.com' in url_lower:
            return 'deezer'
        elif 'music.apple.com' in url_lower:
            return 'apple_music'
        elif 'twitter.com' in url_lower or 'x.com' in url_lower:
            return 'twitter'
        else:
            return 'unknown'
    
    def extract_soundcloud_data(self, url: str) -> Optional[PlatformAnalytics]:
        """Extract SoundCloud analytics: followers from profile, top tracks from /popular-tracks"""
        try:
            # 1. Scrape followers from main profile page
            profile_url = url.split('/popular-tracks')[0].rstrip('/')
            soup_profile = self._get_page_with_selenium(profile_url, wait_seconds=8)
            followers = 0
            username = profile_url.split('/')[-1]
            if soup_profile:
                # Try to extract followers from script tags (JSON)
                scripts = soup_profile.find_all('script')
            for script in scripts:
                if script.string and 'followers_count' in script.string:
                        match = re.search(r'"followers_count":(\d+)', script.string)
                        if match:
                            followers = int(match.group(1))
                            break
                # Fallback: visible text
            if followers == 0:
                    page_text = soup_profile.get_text()
                    match = re.search(r'(\d+[\d,]*)\s+[Ff]ollowers', page_text)
                    if match:
                        followers = int(match.group(1).replace(',', ''))

            # 2. Scrape top 3 unique tracks from /popular-tracks
            pop_url = profile_url + '/popular-tracks'
            soup_pop = self._get_page_with_selenium(pop_url, wait_seconds=12)
            top_tracks = []
            seen_titles = set()
            if soup_pop:
                track_elements = soup_pop.find_all(['li', 'div', 'article'], class_=re.compile(r'soundList__item|trackItem|sound__body|soundList__item', re.IGNORECASE))
                for element in track_elements:
                        try:
                        title_elem = element.find(['a', 'span', 'h2'], class_=re.compile(r'trackTitle|soundTitle__title|sc-link-primary', re.IGNORECASE))
                        if not title_elem:
                            continue
                        track_title = title_elem.get_text(strip=True)
                        if not track_title or len(track_title) < 2 or track_title in seen_titles:
                            continue
                        seen_titles.add(track_title)
                        play_elem = element.find(['span', 'div'], class_=re.compile(r'sc-ministats-plays|playbackCount|play.*count', re.IGNORECASE))
                        play_count = 0
                        if play_elem:
                            play_text = play_elem.get_text(strip=True)
                            match = re.search(r'([\d,]+)', play_text)
                            if match:
                                play_count = int(match.group(1).replace(',', ''))
                        # Try to extract the track URL
                        link_elem = element.find('a', href=True)
                        if link_elem and link_elem['href']:
                            if link_elem['href'].startswith('http'):
                                track_url = link_elem['href']
                            else:
                                track_url = f"https://soundcloud.com{link_elem['href']}"
                        else:
                            track_url = ''
                        top_tracks.append({'title': track_title, 'plays': play_count, 'url': track_url})
                        if len(top_tracks) == 3:
                            break
                    except Exception:
                    continue
            print(f"   👥 Followers: {followers}")
            print("   🔥 Top tracks: " + ', '.join([f"{t['title']} ({t['plays']:,} plays)" for t in top_tracks]))
            return PlatformAnalytics(
                platform='SoundCloud',
                username=username,
                profile_url=profile_url,
                followers=followers,
                total_plays=0,
                recent_posts=[],
                engagement_rate=0.0,
                top_content=top_tracks,
                growth_metrics={
                    'track_count': len(top_tracks),
                },
                last_updated=datetime.now().isoformat()
            )
        except Exception as e:
            print(f"❌ Error extracting SoundCloud data: {e}")
            return None
    
    def extract_tiktok_data(self, url: str) -> Optional[PlatformAnalytics]:
        """Extract TikTok analytics from profile URL"""
        try:
            soup = self._get_page_with_selenium(url, wait_seconds=10)
            if not soup:
                return None
            
            # Extract username from URL
            username = url.split('@')[-1].split('/')[0] if '@' in url else url.split('/')[-1]
            
            # Extract follower count
            followers = 0
            following = 0
            likes = 0
            
            # Method 1: Look in script tags for JSON data
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    script_content = script.string
                    
                    # Look for various TikTok data patterns
                    patterns = {
                        'followers': [r'"followerCount":(\d+)', r'"fans":(\d+)'],
                        'following': [r'"followingCount":(\d+)', r'"following":(\d+)'],
                        'likes': [r'"heartCount":(\d+)', r'"heart":(\d+)', r'"digg":(\d+)']
                    }
                    
                    for metric, pattern_list in patterns.items():
                        for pattern in pattern_list:
                            match = re.search(pattern, script_content)
                            if match:
                                value = int(match.group(1))
                                if metric == 'followers':
                                    followers = max(followers, value)
                                elif metric == 'following':
                                    following = max(following, value)
                                elif metric == 'likes':
                                    likes = max(likes, value)
            
            # Method 2: Look for numbers in visible text
            if followers == 0:
                page_text = soup.get_text()
                
                # Look for follower patterns in text
                follower_patterns = [
                    r'(\d+(?:\.\d+)?[KMB]?)\s*[Ff]ollowers?',
                    r'(\d+(?:\.\d+)?[KMB]?)\s*[Ff]ans?',
                ]
                
                for pattern in follower_patterns:
                    matches = re.findall(pattern, page_text)
                    for match in matches:
                        try:
                            # Convert K, M, B to numbers
                            if 'K' in match:
                                followers = int(float(match.replace('K', '')) * 1000)
                            elif 'M' in match:
                                followers = int(float(match.replace('M', '')) * 1000000)
                            elif 'B' in match:
                                followers = int(float(match.replace('B', '')) * 1000000000)
                            else:
                                followers = int(match)
                            break
                        except:
                            continue
                    if followers > 0:
                        break
            
            # Extract recent videos/posts
            videos = []
            
            # Look for video elements
            video_elements = soup.find_all(['div', 'a'], href=re.compile(r'/video/\d+'))
            
            for video_elem in video_elements[:10]:
                try:
                    # Extract video description/title
                    desc_elem = video_elem.find(['span', 'div'])
                    if desc_elem:
                        description = desc_elem.get_text(strip=True)
                        
                        if description and len(description) > 5:
                            videos.append({
                                'title': description[:100],  # Limit length
                                'url': urljoin(url, video_elem.get('href', '')),
                                'views': 0,  # Hard to extract without individual video pages
                                'likes': 0,
                                'engagement': 0
                            })
                except:
                    continue
            
            print(f"✅ TikTok: {followers} followers, {len(videos)} videos found")
            
            return PlatformAnalytics(
                platform='TikTok',
                username=f"@{username}",
                profile_url=url,
                followers=followers,
                total_plays=0,  # Total views not easily available
                recent_posts=videos,
                engagement_rate=0.0,  # Need individual video data
                top_content=videos[:5],
                growth_metrics={
                    'following': following,
                    'total_hearts': likes,
                    'video_count': len(videos)
                },
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"❌ Error extracting TikTok data: {e}")
            return None
    
    def extract_instagram_data(self, url: str) -> Optional[PlatformAnalytics]:
        """Extract Instagram analytics from profile URL"""
        try:
            soup = self._get_page_with_selenium(url, wait_seconds=8)
            if not soup:
                return None
            
            username = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
            
            # Instagram is heavily protected, look for basic info
            followers = 0
            posts = []
            
            # Look in meta tags or script tags
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'edge_followed_by' in script.string:
                    try:
                        follower_match = re.search(r'"edge_followed_by":\s*\{\s*"count":\s*(\d+)', script.string)
                        if follower_match:
                            followers = int(follower_match.group(1))
                            break
                    except:
                        continue
            
            print(f"✅ Instagram: {followers} followers detected")
            
            return PlatformAnalytics(
                platform='Instagram',
                username=username,
                profile_url=url,
                followers=followers,
                total_plays=0,
                recent_posts=posts,
                engagement_rate=0.0,
                top_content=[],
                growth_metrics={'post_count': len(posts)},
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"❌ Error extracting Instagram data: {e}")
            return None

    def extract_spotify_data(self, url: str) -> Optional[PlatformAnalytics]:
        """Extract Spotify analytics from artist URL"""
        try:
            # First try web scraping
            soup = self._get_page_with_selenium(url, wait_seconds=8)
            if not soup:
                print("❌ Could not load Spotify page with Selenium")
                return None
            
            # Extract artist name and ID from URL
            artist_name = ""
            artist_id = ""
            if '/artist/' in url:
                artist_id = url.split('/artist/')[-1].split('?')[0]
            
            # Look for artist name in page title or meta tags
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                if ' | Spotify' in title_text:
                    artist_name = title_text.replace(' | Spotify', '').strip()
            
            # Extract follower count and monthly listeners
            followers = 0
            monthly_listeners = 0
            
            # Look in page text for listener/follower info
            page_text = soup.get_text()
            
            # Monthly listeners pattern
            listener_patterns = [
                r'(\d+(?:,\d+)*)\s*monthly\s*listeners',
                r'(\d+(?:\.\d+)?[KMB]?)\s*monthly\s*listeners'
            ]
            
            for pattern in listener_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    try:
                        match = matches[0]
                        # Convert K, M, B to numbers
                        if 'K' in match:
                            monthly_listeners = int(float(match.replace('K', '').replace(',', '')) * 1000)
                        elif 'M' in match:
                            monthly_listeners = int(float(match.replace('M', '').replace(',', '')) * 1000000)
                        elif 'B' in match:
                            monthly_listeners = int(float(match.replace('B', '').replace(',', '')) * 1000000000)
                        else:
                            monthly_listeners = int(match.replace(',', ''))
                        break
                    except:
                        continue
            
            # Use monthly listeners as followers if no follower count found
            followers = monthly_listeners
            
            # Extract track information from page
            tracks = []
            
            # Look for track elements in the page
            track_elements = soup.find_all(['div', 'span'], string=re.compile(r'.+'))
            track_titles = []
            
            for elem in track_elements:
                text = elem.get_text(strip=True)
                # Filter for likely track titles (not too short, not too long, no numbers only)
                if (5 < len(text) < 100 and 
                    not text.isdigit() and 
                    not text.lower() in ['spotify', 'play', 'pause', 'skip', 'previous'] and
                    not re.match(r'^\d+:\d+$', text)):  # Not a duration
                    track_titles.append(text)
            
            # Take unique track titles
            seen = set()
            for title in track_titles[:20]:  # Limit to first 20 potential tracks
                if title not in seen and len(title) > 5:
                    seen.add(title)
                    tracks.append({
                        'title': title,
                        'url': url,  # Link back to artist page
                        'plays': 0,  # Spotify doesn't show play counts publicly
                        'type': 'track'
                    })
                if len(tracks) >= 10:  # Limit to 10 tracks
                    break
            
            print(f"✅ Spotify: {followers:,} monthly listeners, {len(tracks)} tracks found")
            
            return PlatformAnalytics(
                platform='Spotify',
                username=artist_name or "Unknown Artist",
                profile_url=url,
                followers=followers,
                total_plays=0,  # Spotify doesn't provide total play counts
                recent_posts=tracks,
                engagement_rate=0.0,
                top_content=tracks[:5],
                growth_metrics={
                    'monthly_listeners': monthly_listeners,
                    'track_count': len(tracks),
                    'artist_id': artist_id
                },
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"❌ Error extracting Spotify data: {e}")
            return None

    def extract_deezer_data(self, url: str) -> Optional[PlatformAnalytics]:
        """Extract Deezer analytics from artist URL"""
        try:
            # Extract artist ID from URL for API fallback
            artist_id = ""
            if '/artist/' in url:
                artist_id = url.split('/artist/')[-1].split('?')[0]
            
            # Try Deezer public API first (no auth required)
            if artist_id and artist_id.isdigit():
                try:
                    api_url = f"https://api.deezer.com/artist/{artist_id}"
                    response = self.session.get(api_url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        artist_name = data.get('name', 'Unknown Artist')
                        fans = data.get('nb_fan', 0)
                        
                        # Get top tracks
                        tracks_url = f"https://api.deezer.com/artist/{artist_id}/top"
                        tracks_response = self.session.get(tracks_url, timeout=10)
                        tracks = []
                        
                        if tracks_response.status_code == 200:
                            tracks_data = tracks_response.json()
                            for track in tracks_data.get('data', [])[:10]:
                                tracks.append({
                                    'title': track.get('title', 'Unknown'),
                                    'url': track.get('link', ''),
                                    'plays': 0,  # Deezer API doesn't provide play counts
                                    'duration': track.get('duration', 0),
                                    'type': 'track'
                                })
                        
                        print(f"✅ Deezer API: {fans:,} fans, {len(tracks)} tracks found")
                        
                        return PlatformAnalytics(
                            platform='Deezer',
                            username=artist_name,
                            profile_url=url,
                            followers=fans,
                            total_plays=0,
                            recent_posts=tracks,
                            engagement_rate=0.0,
                            top_content=tracks[:5],
                            growth_metrics={
                                'fan_count': fans,
                                'track_count': len(tracks),
                                'artist_id': artist_id
                            },
                            last_updated=datetime.now().isoformat()
                        )
                except Exception as api_error:
                    print(f"⚠️ Deezer API failed: {api_error}, trying web scraping...")
            
            # Fallback to web scraping
            soup = self._get_page_with_selenium(url, wait_seconds=8)
            if not soup:
                return None
            
            # Extract artist name
            artist_name = ""
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                if ' - Deezer' in title_text:
                    artist_name = title_text.replace(' - Deezer', '').strip()
            
            # Extract fan count
            fans = 0
            page_text = soup.get_text()
            fan_patterns = [
                r'(\d+(?:,\d+)*)\s*fans?',
                r'(\d+(?:,\d+)*)\s*followers?',
                r'(\d+(?:\.\d+)?[KMB]?)\s*fans?'
            ]
            
            for pattern in fan_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    try:
                        match = matches[0]
                        # Convert K, M, B to numbers
                        if 'K' in match:
                            fans = int(float(match.replace('K', '').replace(',', '')) * 1000)
                        elif 'M' in match:
                            fans = int(float(match.replace('M', '').replace(',', '')) * 1000000)
                        elif 'B' in match:
                            fans = int(float(match.replace('B', '').replace(',', '')) * 1000000000)
                        else:
                            fans = int(match.replace(',', ''))
                        break
                    except:
                        continue
            
            # Extract albums/tracks from page
            albums = []
            # Look for album/track links
            album_elements = soup.find_all('a', href=re.compile(r'/(album|track)/'))
            
            for album_elem in album_elements[:10]:
                try:
                    album_title = album_elem.get_text(strip=True)
                    album_url = urljoin(url, album_elem.get('href'))
                    
                    if album_title and len(album_title) > 2:
                        albums.append({
                            'title': album_title,
                            'url': album_url,
                            'plays': 0,
                            'type': 'album' if '/album/' in album_url else 'track'
                        })
                except:
                    continue
            
            print(f"✅ Deezer: {fans:,} fans, {len(albums)} items found")
            
            return PlatformAnalytics(
                platform='Deezer',
                username=artist_name or "Unknown Artist",
                profile_url=url,
                followers=fans,
                total_plays=0,
                recent_posts=albums,
                engagement_rate=0.0,
                top_content=albums[:5],
                growth_metrics={
                    'fan_count': fans,
                    'album_count': len(albums)
                },
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"❌ Error extracting Deezer data: {e}")
            return None
    
    def extract_from_url(self, url: str) -> Optional[PlatformAnalytics]:
        """Extract analytics from any supported platform URL"""
        platform = self._detect_platform(url)
        
        print(f"🎯 Detected platform: {platform.upper()}")
        
        if platform == 'soundcloud':
            return self.extract_soundcloud_data(url)
        elif platform == 'tiktok':
            return self.extract_tiktok_data(url)
        elif platform == 'instagram':
            return self.extract_instagram_data(url)
        elif platform == 'spotify':
            return self.extract_spotify_data(url)
        elif platform == 'deezer':
            return self.extract_deezer_data(url)
        else:
            print(f"❌ Platform '{platform}' not yet supported")
            return None
    
    def extract_multiple_urls(self, urls: List[str]) -> Dict[str, PlatformAnalytics]:
        """Extract analytics from multiple platform URLs"""
        results = {}
        
        for url in urls:
            try:
                platform = self._detect_platform(url)
                print(f"\n🔍 Processing {platform.upper()}: {url}")
                
                result = self.extract_from_url(url)
                if result:
                    results[platform] = result
                    
                self._random_delay(2, 4)  # Be respectful between requests
                
            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
                continue
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def main():
    """Interactive link-based analytics extractor"""
    print("🎵 BeatWizard Link-Based Analytics Extractor")
    print("=" * 60)
    print("📝 Just paste your profile links - no usernames needed!")
    print("✅ Supported: SoundCloud, TikTok, Instagram, YouTube, Spotify")
    print("")
    
    urls = []
    
    while True:
        url = input("🔗 Paste profile URL (or press Enter when done): ").strip()
        if not url:
            break
        
        if not url.startswith('http'):
            url = 'https://' + url
        
        urls.append(url)
        platform = LinkBasedAnalyticsExtractor()._detect_platform(url)
        print(f"   ✅ Added {platform.upper()} profile")
    
    if not urls:
        print("❌ No URLs provided. Exiting.")
        return
    
    print(f"\n🚀 Processing {len(urls)} profile(s)...")
    
    extractor = LinkBasedAnalyticsExtractor()
    
    try:
        results = extractor.extract_multiple_urls(urls)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 ANALYTICS RESULTS")
        print("=" * 60)
        
        total_followers = 0
        total_content = 0
        
        for platform, data in results.items():
            print(f"\n🎯 {data.platform.upper()}:")
            print(f"   👤 Profile: {data.username}")
            print(f"   👥 Followers: {data.followers:,}")
            print(f"   📝 Content: {len(data.recent_posts)} items")
            print(f"   ▶️ Total Plays: {data.total_plays:,}")
            print(f"   💫 Engagement: {data.engagement_rate:.2%}")
            
            total_followers += data.followers
            total_content += len(data.recent_posts)
            
            if data.top_content:
                print(f"   🔥 Top Content:")
                for i, content in enumerate(data.top_content[:3], 1):
                    title = content.get('title', 'Unknown')
                    plays = content.get('plays', content.get('views', 0))
                    print(f"      {i}. {title[:50]} ({plays:,} plays/views)")
        
        print(f"\n🎉 SUMMARY:")
        print(f"   📱 Platforms: {len(results)}")
        print(f"   👥 Total Followers: {total_followers:,}")
        print(f"   📝 Total Content: {total_content}")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_report_{timestamp}.json"
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'platforms': {k: v.__dict__ for k, v in results.items()},
            'summary': {
                'total_platforms': len(results),
                'total_followers': total_followers,
                'total_content': total_content
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"   💾 Report saved: {filename}")
        
    finally:
        extractor.cleanup()

if __name__ == "__main__":
    main() 