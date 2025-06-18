import librosa
import numpy as np
from textblob import TextBlob
import json
from pathlib import Path

class AudioFeatureExtractor:
    """Extract relevant audio features from beats"""
    
    def __init__(self):
        self.features = {}
    
    def _get_frequency_band_energy(self, y, sr, min_freq, max_freq):
        #Frequency = how fast a sound waves vibrates
        #Lower Frequencies = Lower Pitch
        #Frequency band = a range of frequencies

        #Short Time Fourier Transform on the audio, breaks it down into tiny time slice
        #Figures otu the frequency of each slice and how strong they are
        D = np.abs(librosa.stft(y))
        #2d array with rows as frequencies columns as time steps, and each cell as how strong that frequnecy is at the moemnt
        freqs = librosa.fft_frequencies(sr=sr)

        band_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]

        return np.mean(D[band_indices, :])
    
    def extract_features(self, audio_path):
        """Extract all relevant features from an audio file"""
        # Load the audio file
        try:
            y, sr = librosa.load(audio_path)
        except Exception as e:
            return f"Error loading audio file: {str(e)}"
        
        # Basic tempo and beat features
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Calculate spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # Extract MFCCs for timbre analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate overall energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Bass-specific analysis
        sub_bass_energy = self._get_frequency_band_energy(y, sr, 20, 60)
        bass_energy = self._get_frequency_band_energy(y, sr, 60, 250)
        
        # Onset detection for rhythm analysis
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Rhythm density (onsets per second)
        rhythm_density = len(onset_times) / float(len(y) / sr)
        
        # Beat consistency
        beat_intervals = np.diff(beat_times)
        beat_consistency = 1.0 / np.std(beat_intervals) if len(beat_intervals) > 0 else 0
        
        # Analyze syncopation through onset patterns
        # Get onset strength between beats
        beat_onset_strength = librosa.util.normalize(onset_env)
        syncopation = np.mean(beat_onset_strength[onset_frames])
        
        # Calculate groove features
        groove_patterns = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        groove_strength = np.mean(np.std(groove_patterns, axis=1))
        
        # Get the FFT for frequency analysis
        D = np.abs(librosa.stft(y))
        
        # Additional frequency band analysis for 808s and kicks
        kick_energy = self._get_frequency_band_energy(y, sr, 50, 100)
        snare_energy = self._get_frequency_band_energy(y, sr, 200, 350)
        hihat_energy = self._get_frequency_band_energy(y, sr, 10000, 20000)
        
        # Get onset envelope for rhythm analysis
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Calculate rhythmic regularity
        onset_intervals = np.diff(onset_times)
        rhythmic_regularity = 1.0 / (np.std(onset_intervals) + 1e-6)
        
        # Calculate section changes using spectral novelty
        spec_novelty = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.melspectrogram)
        section_changes = np.mean(spec_novelty > np.median(spec_novelty) * 2)
        
        # Calculate stereo width if stereo file
        if y.ndim > 1 and y.shape[0] == 2:
            stereo_width = np.mean(np.abs(y[0] - y[1]))
        else:
            stereo_width = 0.0
            
        # Update features dict with new features
        self.features.update({
            # Tempo and rhythm features
            'tempo': float(tempo),
            'rhythm_density': float(rhythm_density),
            'beat_consistency': float(beat_consistency),
            'syncopation_score': float(syncopation),
            'groove_strength': float(groove_strength),
            
            # Energy and intensity features
            'average_beat_strength': float(np.mean(librosa.onset.onset_strength(y=y, sr=sr))),
            'energy_mean': float(np.mean(rms)),
            'energy_std': float(np.std(rms)),
            
            # Bass features
            'sub_bass_energy': float(sub_bass_energy),
            'bass_energy': float(bass_energy),
            'bass_to_total_ratio': float(bass_energy / np.mean(np.abs(y)) if np.mean(np.abs(y)) > 0 else 0),
            
            # Spectral features
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_contrast': float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))),
            
            # Percussion features
            'percussive_energy': float(np.mean(np.abs(y_percussive))),
            'harmonic_energy': float(np.mean(np.abs(y_harmonic))),
            'percussion_to_harmonic_ratio': float(np.mean(np.abs(y_percussive)) / np.mean(np.abs(y_harmonic)) if np.mean(np.abs(y_harmonic)) > 0 else 0),
            
            # Timbre features
            'mfcc_means': [float(np.mean(mfcc)) for mfcc in mfccs],
            
            # Specific instrument energies
            'kick_energy': float(kick_energy),
            'snare_energy': float(snare_energy),
            'hihat_energy': float(hihat_energy),
            
            # Advanced rhythm features
            'rhythmic_regularity': float(rhythmic_regularity),
            'section_changes': float(section_changes),
            
            # Production features
            'stereo_width': float(stereo_width),
            
            # Ratios between elements
            'kick_to_snare_ratio': float(kick_energy / (snare_energy + 1e-6)),
            'hihat_to_kick_ratio': float(hihat_energy / (kick_energy + 1e-6))
        })
        
        return self.features

    def get_features_description(self):
        """Return a human-readable description of the features with raw numbers"""
        if not self.features:
            return "No features extracted yet"
            
        description = []
        
        description.append(f"=== Rhythm Analysis ===")
        description.append(f"Tempo: {self.features['tempo']:.1f} BPM")
        description.append(f"Rhythm Density: {self.features['rhythm_density']:.2f} hits/sec")
        description.append(f"Beat Consistency: {self.features['beat_consistency']:.2f}")
        description.append(f"Groove Strength: {self.features['groove_strength']:.3f}")
        description.append(f"Syncopation Score: {self.features['syncopation_score']:.3f}")
        
        description.append(f"\n=== Bass Analysis ===")
        description.append(f"Sub-bass Energy (20-60Hz): {self.features['sub_bass_energy']:.3f}")
        description.append(f"Bass Energy (60-250Hz): {self.features['bass_energy']:.3f}")
        description.append(f"Bass/Total Ratio: {self.features['bass_to_total_ratio']:.3%}")
        
        description.append(f"\n=== Energy & Dynamics ===")
        description.append(f"Average Energy: {self.features['energy_mean']:.3f}")
        description.append(f"Energy Variation: {self.features['energy_std']:.3f}")
        description.append(f"Average Beat Strength: {self.features['average_beat_strength']:.3f}")
        
        description.append(f"\n=== Spectral Features ===")
        description.append(f"Spectral Centroid: {self.features['spectral_centroid_mean']:.1f} Hz")
        description.append(f"Spectral Rolloff: {self.features['spectral_rolloff_mean']:.1f} Hz")
        description.append(f"Spectral Bandwidth: {self.features['spectral_bandwidth_mean']:.1f} Hz")
        description.append(f"Spectral Contrast: {self.features['spectral_contrast']:.3f}")
        
        description.append(f"\n=== Balance Features ===")
        description.append(f"Percussive Energy: {self.features['percussive_energy']:.3f}")
        description.append(f"Harmonic Energy: {self.features['harmonic_energy']:.3f}")
        description.append(f"Percussion/Harmonic Ratio: {self.features['percussion_to_harmonic_ratio']:.3f}")
        
        description.append(f"\n=== Timbre Features ===")
        for i, mfcc in enumerate(self.features['mfcc_means']):
            description.append(f"MFCC {i+1}: {mfcc:.3f}")
        
        return "\n".join(description)

class SentimentAnalyzer:
    """Analyze sentiment and key descriptors from beat descriptions"""
    
    def __init__(self):
        self.analysis = {}
        # Ensure NLTK data is downloaded
        try:
            import nltk
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
    
    def analyze_text(self, text):
        """Analyze the sentiment and extract key descriptors from text"""
        try:
            blob = TextBlob(text)
            
            # Get sentiment
            sentiment = blob.sentiment.polarity
            
            # Extract descriptive words (adjectives)
            try:
                descriptors = [word for (word, tag) in blob.tags if tag.startswith('JJ')]
            except:
                # Fallback if tagging fails
                descriptors = [word for word in blob.words if len(word) > 2]  # Simple fallback
            
            self.analysis = {
                'sentiment': sentiment,
                'descriptors': descriptors,
                'raw_text': text
            }
            
            return self.analysis
            
        except Exception as e:
            print(f"Warning: Text analysis failed: {e}")
            return {
                'sentiment': 0.0,
                'descriptors': [],
                'raw_text': text
            }
    
    def get_sentiment_description(self):
        """Return a human-readable description of the sentiment analysis"""
        if not self.analysis:
            return "No text analyzed yet"
            
        mood = "positive" if self.analysis['sentiment'] > 0 else "negative" if self.analysis['sentiment'] < 0 else "neutral"
        confidence = "strongly" if abs(self.analysis['sentiment']) > 0.5 else "slightly"
        
        description = []
        description.append(f"Mood: {confidence} {mood}")
        description.append(f"Key descriptors: {', '.join(self.analysis['descriptors'])}")
        
        return "\n".join(description)

def main():
    """Main function to run the beat analyzer"""
    print("\n=== Beat Analyzer Tool ===\n")
    
    # Get audio file path
    audio_path = input("Enter the path to your beat file (WAV/MP3): ").strip()
    
    # Check if file exists
    if not Path(audio_path).is_file():
        print(f"Error: File '{audio_path}' not found!")
        return
    
    # Get beat description
    description = input("\nGive a brief description of this beat (style, mood, etc.): ").strip()
    
    print("\nAnalyzing...")
    
    # Extract audio features
    extractor = AudioFeatureExtractor()
    features = extractor.extract_features(audio_path)
    
    # Analyze description
    analyzer = SentimentAnalyzer()
    sentiment = analyzer.analyze_text(description)
    
    # Display results
    print("\n=== Audio Features ===")
    print(extractor.get_features_description())
    
    print("\n=== Description Analysis ===")
    print(analyzer.get_sentiment_description())
    
    # Save results to JSON
    results = {
        'audio_features': features,
        'text_analysis': sentiment
    }
    
    output_file = Path(audio_path).stem + '_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull analysis saved to: {output_file}")

if __name__ == "__main__":
    main() 