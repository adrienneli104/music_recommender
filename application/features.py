import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd

def get_mean_features(playlist):
    n = float(len(playlist))
    features = {'danceability' : 0,
                'loudness' : 0,
                'speechiness' : 0,
                'acousticness' : 0,
                'instrumentalness' : 0,
                'liveness' : 0,
                'valence' : 0,
                'tempo' : 0 }
    for index, row in playlist.iterrows():
        features['danceability'] += row['danceability']
        features['loudness'] += row['loudness']
        features['speechiness'] += row['speechiness']
        features['acousticness'] += row['acousticness']
        features['instrumentalness'] += row['instrumentalness']
        features['liveness'] += row['liveness']
        features['valence'] += row['valence']
        features['tempo'] += row['tempo']
    return [round(x/n, 2) for x in features.values()]

def extract(URL):
    client_id = "5356afb958c84e71a2c37c43e2a2cbf2" 
    client_secret = "83e531491e9c458ba658ac30c4c56bc0"

    #use the clint secret and id details
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id,client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # the URI is split by ':' to get the username and playlist ID
    playlist_id = URL.split("/")[4].split("?")[0]
    playlist_tracks_data = sp.playlist_tracks(playlist_id)

    #lists that will be filled in with features
    playlist_tracks_id = []

    #go through the dictionary to extract the data
    for track in playlist_tracks_data['items']:
        playlist_tracks_id.append(track['track']['id'])

    #create a dataframe
    features = sp.audio_features(playlist_tracks_id)
    features_df = pd.DataFrame(data=features, columns=features[0].keys())
    features_df = features_df[['id', 'danceability', 'loudness', 
                                'speechiness', 'acousticness', 'instrumentalness', 
                                'liveness', 'valence', 'tempo']]
    features_df.rename(columns={'id': 'track_id'}, inplace=True)
    return features_df



