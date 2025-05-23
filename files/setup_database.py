import sqlite3
import json
import random
import os

# Define the features we will store and use for similarity
# These should match keys in the 'audio_features' part of the JSON
FEATURES_TO_STORE = [
    "tempo", "energy_mean", "rhythm_density", "beat_consistency", 
    "syncopation_score", "spectral_centroid_mean", "spectral_rolloff_mean", 
    "spectral_bandwidth_mean", "percussive_energy", "harmonic_energy"
]

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    # Remove the db file if it exists to start fresh each time
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing database: {db_file}")
        
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Database created/connected: {db_file}")
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        print("Table 'songs' created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

def insert_song(conn, song_data_tuple):
    """
    Create a new song into the songs table
    :param conn: Connection object
    :param song_data_tuple: A tuple containing song data (title, followed by feature values, then danceability_mock)
    :return: song id
    """
    # Dynamically create the SQL query based on FEATURES_TO_STORE
    feature_columns = ", ".join(FEATURES_TO_STORE)
    placeholders = ", ".join(["?"] * (len(FEATURES_TO_STORE) + 2)) # +2 for title and danceability_mock
    
    sql = f'''INSERT INTO songs(title, {feature_columns}, danceability_mock)
              VALUES({placeholders})'''
    # Example: INSERT INTO songs(title, tempo, energy_mean, ..., danceability_mock) VALUES(?, ?, ?, ..., ?)

    cur = conn.cursor()
    try:
        cur.execute(sql, song_data_tuple)
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        print(f"Error inserting song {song_data_tuple[0]}: {e}")
        print(f"SQL: {sql}")
        print(f"Data: {song_data_tuple}")
        return None

def generate_variant_features(base_features_dict):
    """ Generates slightly varied features based on a base dictionary. """
    variant = {}
    for key, value in base_features_dict.items():
        if isinstance(value, (int, float)):
            # Apply a small random variation (e.g., +/- 10%)
            variation_factor = random.uniform(0.9, 1.1)
            variant[key] = value * variation_factor
        else:
            variant[key] = value # Keep non-numeric as is (though we mostly use numeric)
    return variant

def main():
    database_filename = "song_database.db"
    # Construct the full path to the database file within the 'files' directory
    # Assuming this script is run from the workspace root (e.g., /Users/pamehta/BeatWizard)
    # and the database should also be in 'files/'
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Should be /Users/pamehta/BeatWizard/files
    database_filepath = os.path.join(script_dir, database_filename)

    # Dynamically create the table schema based on FEATURES_TO_STORE
    feature_definitions = ", ".join([f"{feature} REAL" for feature in FEATURES_TO_STORE])
    sql_create_songs_table = f""" CREATE TABLE IF NOT EXISTS songs (
                                        id INTEGER PRIMARY KEY,
                                        title TEXT NOT NULL,
                                        {feature_definitions},
                                        danceability_mock REAL 
                                    ); """
    # Example: tempo REAL, energy_mean REAL, ...

    conn = create_connection(database_filepath)

    if conn is not None:
        create_table(conn, sql_create_songs_table)

        all_songs_data = []

        # 1. Load the base song from draketorytypebeat_analysis.json
        base_song_json_path = os.path.join(script_dir, "draketorytypebeat_analysis.json")
        
        try:
            with open(base_song_json_path, 'r') as f:
                base_song_full_data = json.load(f)
            base_audio_features = base_song_full_data.get("audio_features", {})
            
            # Extract only the features we want to store
            base_song_feature_values = [base_audio_features.get(f) for f in FEATURES_TO_STORE]
            
            # Check if all features were found
            if None in base_song_feature_values:
                print(f"Warning: Not all features found in {base_song_json_path}. Check FEATURES_TO_STORE.")
                # Filter out None values if any feature is missing, or handle as error
                # For now, let's ensure we have the right number of features or skip
                if len([v for v in base_song_feature_values if v is not None]) != len(FEATURES_TO_STORE):
                    print(f"Error: Could not extract all required features from {base_song_json_path}. Aborting population.")
                    conn.close()
                    return
            
            # (title, feature1, feature2, ..., danceability_mock)
            draketory_data = ("Draketory Type Beat (Original)",) + tuple(base_song_feature_values) + (random.uniform(0.3, 0.9),)
            all_songs_data.append(draketory_data)

        except FileNotFoundError:
            print(f"Error: {base_song_json_path} not found. Cannot populate base song.")
            conn.close()
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {base_song_json_path}.")
            conn.close()
            return
        except Exception as e:
            print(f"An unexpected error occurred while processing base song: {e}")
            conn.close()
            return

        # 2. Generate 9 more variant songs
        base_features_for_variants = {feat_name: base_audio_features.get(feat_name) for feat_name in FEATURES_TO_STORE if base_audio_features.get(feat_name) is not None}

        if len(base_features_for_variants) != len(FEATURES_TO_STORE):
            print("Error: Base features for generating variants are incomplete due to missing keys in the JSON. Cannot generate variants.")
        else:
            for i in range(9):
                variant_audio_features = generate_variant_features(base_features_for_variants.copy()) # Pass a copy
                # Extract values in the correct order
                variant_feature_values = [variant_audio_features.get(f) for f in FEATURES_TO_STORE]
                
                song_tuple = (f"Variant Song {i+1}",) + tuple(variant_feature_values) + (random.uniform(0.3, 0.9),) # Add mock danceability
                all_songs_data.append(song_tuple)

        print(f"Attempting to insert {len(all_songs_data)} songs...")
        for song_data_tuple in all_songs_data:
             # Check length consistency
            expected_length = 1 + len(FEATURES_TO_STORE) + 1 # 1 for title, 1 for danceability_mock
            if len(song_data_tuple) == expected_length:
                insert_song(conn, song_data_tuple)
            else:
                print(f"Skipping song due to incorrect data length: {song_data_tuple[0]}. Expected {expected_length} items, got {len(song_data_tuple)}.")
                print(f"Data: {song_data_tuple}")

        print(f"Database setup complete. '{database_filepath}' created and populated with {len(all_songs_data)} songs.")
        conn.close()
    else:
        print("Error! Cannot create the database connection.")

if __name__ == '__main__':
    main() 