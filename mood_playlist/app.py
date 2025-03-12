from flask import Flask, request, render_template
from transformers import pipeline
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import nltk
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

# Download NLTK resources
nltk.download('punkt')

# Flask app setup
app = Flask(__name__)

# Load the Emotion Detection Model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

# Spotify API Configuration
spotify = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID, 
    client_secret=SPOTIFY_CLIENT_SECRET
))

# Mood-to-Genre Mapping
def mood_to_genre(emotion):
    mood_map = {
        "joy": ["pop", "dance"],
        "sadness": ["acoustic", "instrumental"],
        "anger": ["rock", "rap"],
        "fear": ["chill", "ambient"],
        "love": ["romantic", "ballad"],
        "neutral": ["pop"]
    }
    return mood_map.get(emotion.lower(), ["pop"])  # Default to "pop" if emotion is not mapped

# Fetch Songs from Spotify
def fetch_songs(genres):
    tracks = []
    for genre in genres:
        try:
            results = spotify.search(q=f"genre:{genre}", type="track", limit=5)
            print(f"Spotify API results for genre '{genre}': {results}")  # Debug print
            if "tracks" in results and "items" in results["tracks"]:
                for track in results["tracks"]["items"]:
                    tracks.append({
                        "name": track.get("name", "Unknown"),
                        "artist": track["artists"][0].get("name", "Unknown") if track.get("artists") else "Unknown",
                        "url": track["external_urls"].get("spotify", "#") if track.get("external_urls") else "#"
                    })
        except Exception as e:
            print(f"Error fetching songs for genre {genre}: {e}")
    return tracks

# Main Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        journal_entry = request.form.get('entry', '').strip()
        if not journal_entry:
            return render_template('index.html', error="Please enter a valid journal entry.")

        try:
            # Detect emotion from journal entry
            emotion_result = emotion_classifier(journal_entry)[0]['label']
            print(f"Emotion detected: {emotion_result}")  # Debug print
            genres = mood_to_genre(emotion_result)
            playlist = fetch_songs(genres)

            return render_template('index.html', emotion=emotion_result, playlist=playlist)
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {e}")

    return render_template('index.html')

# Run the App
if __name__ == '__main__':
    app.run(debug=True)
