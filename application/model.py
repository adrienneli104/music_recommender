import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist
import re

'''
  Description: Cleans data to only exclude categorical variables
    
  Input Param:
  -d: song data

  Output Param:
  -d: song data without categorical variables
'''
def get_relevant_cols(d):
    return d.drop(['track_id', 'track_genre', 'album_name','track_name', 'explicit', 'artists', 'mode', 'key',  'time_signature', 'duration_ms', 'popularity', 'energy'], axis=1)
'''
  Description: Returns number of clusters, frequency and percentage for each
  given model prediction for user playlist

  Input Param:
  -predictions: model prediction for user playlist
  -top_n: number of cluster classes

  Output Param:
  -cluster_num: minimum between top_n and unique values
  -freq_clusters: clusters sorted by increasing freq
  -freq_percent: percentage that each cluster should be used in the recommendation
'''
def get_freq_clusters(predictions, top_n):
    unique_vals, freq = np.unique(predictions, return_counts=True)
    cluster_num = min(unique_vals.shape[0], top_n)
    sorted_i = np.argsort(freq)[::-1]
    sorted_freq = freq[sorted_i]
    freq_sum = np.sum(sorted_freq[:cluster_num])
    freq_clusters = unique_vals[sorted_i]
    freq_percent = [(i / freq_sum) for i in sorted_freq[:cluster_num]]
    return cluster_num, freq_clusters, freq_percent

'''
  Description: Given a spotify playlist, applies K-Means model and returns
  song recommendations that have similar audio features
    
  Input Param:
  -playlist: user playlist, list of track_id
  -data: dataset with all songs
  -scaler: fitted scaler
  -model: k-cluster model
  -recNum: number of song recommendations
  -top_n: number of cluster classes

  Output Param:
  -recs_id: song recommendations
'''
def make_recommendations(playlist, data, scaler, model, recNum=5, top_n=3):
    # Transform and predict
    songs = playlist['track_id'].values.tolist()
    X = playlist.drop(['track_id'], axis=1)
    transformed_X = scaler.fit_transform(X)
    predictions = model.predict(transformed_X)

    dataset = get_relevant_cols(data)
    dataset = scaler.transform(dataset)

    # Get most frequent cluster classes from user input
    cluster_num, freq_clusters, freq_percent = get_freq_clusters(predictions, top_n)

    recs_id = pd.DataFrame(columns=['track_id', 'similarity'])
    recs = pd.DataFrame(columns=data.columns)
    for i in range(cluster_num):
        # Number of recommendations from given cluster
        rec_num = round(freq_percent[i] * recNum)
        cur_cluster = freq_clusters[i]

        # Create mean vector and calculate similarity
        pos = np.where(predictions == cur_cluster)[0]
        cluster_songs = transformed_X[pos, :]
        mean_song = np.mean(cluster_songs, axis=0)
        similarity = cdist(np.reshape(mean_song, (1,-1)), dataset)

        # Sort to get similar songs
        similarity_s = pd.Series(similarity.flatten(), name='similarity')
        similar_songs = pd.concat([data['track_id'].reset_index(drop=True), similarity_s.reset_index(drop=True)], axis=1)
        recs_id = recs_id._append(similar_songs)

        # Remove songs from user_songs list
        similar_songs = similar_songs[~(similar_songs['track_id'].isin(songs))]
        similar_songs = similar_songs.sort_values(by='similarity', ascending=True).reset_index(drop=True)

    recs_id = recs_id.reset_index(drop=True)
    return recs_id.loc[:recNum-1]
