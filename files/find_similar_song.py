import sqlite3
import argparse
import json
import math
import os
import numpy as np # For vector operations

# Define the features we will use for similarity
# This list MUST be consistent with setup_database.py
FEATURES_FOR_SIMILARITY = [
    "tempo", "energy_mean", "rhythm_density", "beat_consistency", 
    "syncopation_score", "spectral_centroid_mean", "spectral_rolloff_mean", 
    "spectral_bandwidth_mean", "percussive_energy", "harmonic_energy"
]

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database {db_file}: {e}")
    return conn

def get_all_songs_features(conn):
    """ Query all rows and relevant features in the songs table """
    cur = conn.cursor()
    try:
        # Dynamically select the feature columns
        feature_columns_str = ", ".join(FEATURES_FOR_SIMILARITY)
        cur.execute(f"SELECT id, title, {feature_columns_str} FROM songs")
        rows = cur.fetchall()
        # Get column names to create a list of dictionaries
        column_names = [description[0] for description in cur.description]
        
        songs_with_features = []
        for row in rows:
            song_dict = dict(zip(column_names, row))
            songs_with_features.append(song_dict)
        return songs_with_features
    except sqlite3.Error as e:
        print(f"Error fetching songs: {e}")
        return []

def extract_features_from_json(json_filepath):
    """ Extracts relevant audio features from the input JSON file. """
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        
        audio_features = data.get("audio_features", {})
        
        extracted_values = []
        for feature_name in FEATURES_FOR_SIMILARITY:
            value = audio_features.get(feature_name)
            if value is None:
                print(f"Warning: Feature '{feature_name}' not found in input JSON '{json_filepath}'. Using 0 as default.")
                extracted_values.append(0) # Or handle more gracefully, e.g., raise error or skip
            else:
                extracted_values.append(value)
        return extracted_values
        
    except FileNotFoundError:
        print(f"Error: Input JSON file not found: {json_filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing input JSON: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """ Calculate cosine similarity between two vectors """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0 # Avoid division by zero; vectors with zero magnitude have zero similarity to anything else
    return dot_product / (norm_vec1 * norm_vec2)

def normalize_features(db_feature_vectors_list, input_song_features_list):
    """ Normalizes features using Min-Max scaling based on dataset values. """
    if not db_feature_vectors_list:
        print("Warning: Database feature vectors list is empty. Cannot perform robust normalization.")
        # If input is valid, return it as is (or normalized against itself resulting in zeros if that's desired)
        # and an empty list for db songs. For now, returning as is.
        if input_song_features_list and len(input_song_features_list) == len(FEATURES_FOR_SIMILARITY):
             # Returning the input song un-normalized, and an empty list for DB songs.
             # This scenario might need more specific handling depending on desired behavior.
            return [], input_song_features_list 
        return [], [] # Both empty or input invalid

    num_features = len(FEATURES_FOR_SIMILARITY)
    # Ensure all vectors in db_feature_vectors_list have the correct number of features
    for i, vec in enumerate(db_feature_vectors_list):
        if len(vec) != num_features:
            print(f"Warning: DB song vector at index {i} has {len(vec)} features, expected {num_features}. Skipping or handling.")
            # Potentially remove or fix this vector. For now, this might lead to np.array error if dimensions mismatch.
            # A robust way is to filter out such vectors before this stage.
            # For this fix, we assume vectors are correctly pre-processed to have the right length.
    if len(input_song_features_list) != num_features:
        print(f"Warning: Input song vector has {len(input_song_features_list)} features, expected {num_features}.")
        # Handle error: cannot proceed with mismatched feature numbers for normalization across dataset.
        return [], [] 

    combined_features = []
    combined_features.extend(db_feature_vectors_list) # db_feature_vectors_list is already a list of lists
    combined_features.append(input_song_features_list)
    
    try:
        combined_features_np = np.array(combined_features, dtype=float)
    except ValueError as e:
        print(f"Error creating numpy array for normalization, likely due to inconsistent feature vector lengths: {e}")
        # This can happen if a feature vector has a different number of elements than others.
        # Earlier checks should prevent this, but it's a safeguard.
        return [], []
    
    min_vals = np.min(combined_features_np, axis=0)
    max_vals = np.max(combined_features_np, axis=0)
    
    range_vals = max_vals - min_vals
    
    # Avoid division by zero if a feature is constant across all songs + input
    # If range is 0, the normalized value will be 0 (or 0.5 if min=max)
    # For features that are constant, they won't contribute to distinguishing songs,
    # so setting their normalized value to 0 or a fixed point is reasonable.
    range_vals[range_vals == 0] = 1 # Prevent division by zero, effectively making (val - min) / 1

    normalized_all = (combined_features_np - min_vals) / range_vals
    
    normalized_db_songs = normalized_all[:-1].tolist() # All but the last one (input song)
    normalized_input_song = normalized_all[-1].tolist() # Only the last one
    
    return normalized_db_songs, normalized_input_song


def main():
    parser = argparse.ArgumentParser(description="Find the most similar song in the database using cosine similarity on multiple audio features.")
    parser.add_argument("input_json", type=str, help="Path to the JSON file containing audio features of the input song.")
    
    args = parser.parse_args()

    input_song_raw_features = extract_features_from_json(args.input_json)
    if input_song_raw_features is None:
        return # Error message already printed

    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_filepath = os.path.join(script_dir, "song_database.db")

    conn = create_connection(database_filepath)
    if conn is None:
        return

    db_songs_with_features = get_all_songs_features(conn)
    conn.close()

    if not db_songs_with_features:
        print("No songs found in the database. Please run setup_database.py first.")
        return

    # Normalize features
    # First, extract just the feature vectors from the db_songs_with_features list of dicts
    db_feature_vectors_raw = []
    for song_dict in db_songs_with_features:
        vector = [song_dict.get(f_name) for f_name in FEATURES_FOR_SIMILARITY]
        # Basic check for None values (should ideally be handled more robustly or prevented at DB insertion)
        if any(v is None for v in vector):
            print(f"Warning: Song '{song_dict.get('title', 'Unknown')}' has missing features. Skipping normalization for this song or using 0.")
            # Option 1: Skip this song (might be too strict)
            # Option 2: Replace None with 0 or mean (done in normalize_features if default used)
            vector = [v if v is not None else 0 for v in vector] # Replace None with 0 for normalization
        db_feature_vectors_raw.append(vector)
        
    normalized_db_feature_vectors, normalized_input_song_vector = normalize_features(db_feature_vectors_raw, input_song_raw_features)

    if not normalized_db_feature_vectors: # Check if normalization failed or resulted in empty
        print("Could not normalize features. Aborting similarity comparison.")
        return

    most_similar_song_title = None
    highest_similarity_score = -1 # Cosine similarity is between -1 and 1; closer to 1 is more similar

    for i, db_song_dict in enumerate(db_songs_with_features):
        db_song_title = db_song_dict["title"]
        
        # Check if we have a corresponding normalized vector (e.g. if a song was skipped due to missing raw data)
        if i >= len(normalized_db_feature_vectors):
            print(f"Skipping similarity for '{db_song_title}' due to normalization issues.")
            continue
            
        normalized_db_song_vector = normalized_db_feature_vectors[i]
        
        try:
            similarity = cosine_similarity(normalized_input_song_vector, normalized_db_song_vector)
        except Exception as e:
            print(f"Error calculating cosine similarity for song '{db_song_title}': {e}")
            continue
            
        if similarity > highest_similarity_score:
            highest_similarity_score = similarity
            most_similar_song_title = db_song_title

    if most_similar_song_title:
        print(f"The most similar song to your input (from '{args.input_json}') is: '{most_similar_song_title}' (Cosine Similarity: {highest_similarity_score:.4f})")
    else:
        print("Could not determine the most similar song. This might happen if the database is empty or songs have incomplete/unnormalizable data.")

if __name__ == '__main__':
    main() 