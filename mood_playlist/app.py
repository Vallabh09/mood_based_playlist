# Mood-Based Playlist Generator: Full Project Code

### **1. Initial Setup**
# Install Required Libraries
# pip install flask transformers torch spotipy nltk sklearn pandas

### **2. Project Structure**
# - app.py (Main Backend Logic)
# - models/ (Holds Emotion Detection Models)
# - static/ (Frontend Static Files: JS, CSS)
# - templates/ (HTML Files for Frontend)
# - config.py (Spotify API Keys)

# Step 1: Import Libraries
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('punkt')

# Step 2: Flask App Setup
app = Flask(__name__)

# Step 3: Load Emotion Detection Model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)

# Step 4: Spotify API Configuration
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
spotify = Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))

# Step 5: Mood-to-Music Mapping
def mood_to_genre(emotion):
    mood_map = {
        "joy": ["pop", "dance"],
        "sadness": ["acoustic", "instrumental"],
        "anger": ["rock", "rap"],
        "fear": ["chill", "ambient"],
        "love": ["romantic", "ballad"],
    }
    return mood_map.get(emotion.lower(), ["pop"])

# Step 6: Music Recommendation Engine
def fetch_songs(genres):
    tracks = []
    for genre in genres:
        results = spotify.search(q=f"genre:{genre}", type="track", limit=5)
        for track in results["tracks"]["items"]:
            tracks.append({
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "url": track["external_urls"]["spotify"]
            })
    return tracks

# Step 7: Main Backend Logic
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        journal_entry = request.form['entry']
        emotion = emotion_classifier(journal_entry)[0]['label']
        genres = mood_to_genre(emotion)
        playlist = fetch_songs(genres)

        return render_template('index.html', emotion=emotion, playlist=playlist)
    return render_template('index.html')

# Step 8: Run the App
if __name__ == '__main__':
    app.run(debug=True)
