from application import app
from flask import Flask, render_template, request
from application.features import *
from application.model import *
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

data = pd.read_csv("data/all_data.csv")
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = StandardScaler()

@app.route("/")
def home():
   return render_template('home.html')

@app.route("/about")
def about():
   return render_template('about.html')

@app.route('/recommend', methods=['POST'])
def recommend():
   # Get playlist data from user input (spotify playlist link)
   URL = request.form['URL']
   playlist = extract(URL)
   # Get number of song recommendations from user input
   number_of_recs = int(request.form['number-of-recs'])
   # Get recommendations
   model_result = make_recommendations(playlist, data, scaler, model, number_of_recs)
   rec_ids = list(model_result['track_id'])
   song_data = data[data['track_id'].isin(rec_ids)]
   song_data['similarity'] = model_result['similarity']
   song_recomendations = []
   # Store spotify song link, track name, artist name, and similarity score
   for i in range(number_of_recs):
      recommendation = song_data.loc[[i]]
      link = "https://open.spotify.com/track/" + recommendation['track_id'].values[0]
      track_name = recommendation['track_name'].values[0].capitalize()
      first, last = recommendation['artists'].values[0].split(" ")
      artist_name = first.capitalize() + " " + last.capitalize()
      sim = recommendation['similarity'].values[0]
      song_recomendations.append([track_name, artist_name, link, sim])
   return render_template('results.html',songs= song_recomendations)