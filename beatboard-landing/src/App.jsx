import React, { useState, useEffect } from 'react';
import './App.css';

const SUPABASE_URL = 'https://alncoqxfrhpnmbyzxwwz.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFsbmNvcXhmcmhwbm1ieXp4d3d6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc5NTkzOTYsImV4cCI6MjA2MzUzNTM5Nn0.3EbTCpYBY1xeWOFMFFKNq07nL7qIIA_om3l7kF__CtY';

const carouselPhrases = [
  { before: 'Wave goodbye to', highlight: ' missed opportunities', after: '' },
  { before: 'No more', highlight: ' creative blocks', after: '' },
  { before: 'Find your next', highlight: ' perfect collab', after: '' },
  { before: 'Break through the', highlight: ' noise', after: '' },
  { before: 'Unlock', highlight: ' real connections', after: '' },
  { before: 'Flip the', highlight: ' 96%', after: ' and get heard' },
];

function TextCarousel() {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % carouselPhrases.length);
    }, 2200);
    return () => clearInterval(timer);
  }, []);
  const phrase = carouselPhrases[index];
  return (
    <div className="text-carousel">
      <span>{phrase.before} </span>
      <span className="carousel-highlight">{phrase.highlight}</span>
      <span>{phrase.after}</span>
    </div>
  );
}

function App() {
  const [email, setEmail] = useState('');
  const [userType, setUserType] = useState('artist');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess(false);
    const safeUserType = userType && (userType === 'artist' || userType === 'producer') ? userType : 'artist';
    try {
      const res = await fetch(`${SUPABASE_URL}/rest/v1/interest_form`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'apikey': SUPABASE_ANON_KEY,
          'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
          'Prefer': 'return=minimal',
        },
        body: JSON.stringify({ email, user_type: safeUserType }),
      });
      if (res.ok) {
        setSuccess(true);
        setEmail('');
        setUserType('artist');
      } else {
        const data = await res.json();
        setError(data.message || 'Submission failed.');
      }
    } catch (err) {
      setError('Network error.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-bg">
      {/* Navbar */}
      <nav className="navbar">
        <div className="navbar-brand">BeatBoard</div>
        <div className="navbar-links">
          <a href="#features">Features</a>
          <a href="#waitlist">Join Waitlist</a>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="hero">
        <h1 className="hero-title">BeatBoard</h1>
        <TextCarousel />
        <a href="#waitlist" className="cta-btn">Join the Movement</a>
      </header>
      <section className="waitlist-section" id="waitlist">
        <div className="waitlist-card">
          <h2>Join the Waitlist</h2>
          <p>Be among the first to experience the future of music collaboration.</p>
          <form onSubmit={handleSubmit}>
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
              placeholder="Enter your email"
            />
            <label>I am a</label>
            <select value={userType} onChange={e => setUserType(e.target.value)}>
              <option value="artist">Artist</option>
              <option value="producer">Producer</option>
            </select>
            <button type="submit" disabled={loading}>
              {loading ? 'Submitting...' : 'Join Waitlist'}
            </button>
            {success && <div className="success-msg">Welcome to BeatBoard! We'll be in touch soon.</div>}
            {error && <div className="error-msg">{error}</div>}
          </form>
        </div>
      </section>

      {/* Key Features Section */}
      <section className="info-section" id="features">
        <h2>Why BeatBoard?</h2>
        <ul className="features-list">
          <li>
            <span className="feature-title">🎯 Smart Matching</span>
            <span className="feature-desc">Our AI-powered platform connects you with the perfect collaborators based on your style, goals, and vision.</span>
          </li>
          <li>
            <span className="feature-title">🚀 Growth Engine</span>
            <span className="feature-desc">Access tools and insights designed to help you break through the 96% barrier and reach your audience.</span>
          </li>
          <li>
            <span className="feature-title">🤝 Real Connections</span>
            <span className="feature-desc">Build meaningful relationships with artists and producers who share your passion and drive.</span>
          </li>
          <li>
            <span className="feature-title">📊 Analytics & Insights</span>
            <span className="feature-desc">Track your growth, understand your audience, and make data-driven decisions to accelerate your career.</span>
          </li>
          <li>
            <span className="feature-title">🎵 Quality Control</span>
            <span className="feature-desc">Our platform ensures high-quality collaborations through verified profiles and transparent feedback systems.</span>
          </li>
          <li>
            <span className="feature-title">💫 Community First</span>
            <span className="feature-desc">Join a thriving community of creators who support, inspire, and elevate each other's work.</span>
          </li>
        </ul>
      </section>

      {/* Platform Features Highlight Section */}
      <section className="features-highlight">
        <h2>What Makes BeatBoard Different?</h2>
        <div className="features-highlight-grid">
          <div className="feature-highlight-card">
            <div className="feature-highlight-icon">🔥</div>
            <div className="feature-highlight-title">Momentum Score</div>
            <div className="feature-highlight-desc">
              We calculate your overall momentum from all your platforms—TikTok, Instagram, YouTube, Spotify, SoundCloud, and more—so you always know where you stand.
            </div>
            <div className="feature-img-wrapper">
              <img className="feature-img" src="/images/momentum-score.png" alt="Momentum Score" />
            </div>
          </div>
          <div className="feature-highlight-card">
            <div className="feature-highlight-icon">📈</div>
            <div className="feature-highlight-title">Platform Performance</div>
            <div className="feature-highlight-desc">
              Analyze your most popular tracks to quantify your biggest songs and get data-driven recommendations to help you create your next hit.
            </div>
            <div className="feature-img-wrapper">
              <img className="feature-img" src="/images/different-platforms.png" alt="Platform Performance" />
            </div>
          </div>
          <div className="feature-highlight-card">
            <div className="feature-highlight-icon">🃏</div>
            <div className="feature-highlight-title">Action Cards</div>
            <div className="feature-highlight-desc">
              Swipe through actionable, gamified cards that give you personalized insights and next steps to grow your audience and career.
            </div>
            <div className="feature-img-wrapper">
              <img className="feature-img" src="/images/actionable-insights.png" alt="Action Cards" />
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        &copy; {new Date().getFullYear()} BeatBoard. All rights reserved.
      </footer>
    </div>
  );
}

export default App;
