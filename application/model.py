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
    
  -param[in] d: song data

  -param[out] d: song data without categorical variables
'''
def get_relevant_cols(d):
    return d.drop(['track_id', 'track_genre', 'album_name','track_name', 'explicit', 'artists', 'mode', 'key',  'time_signature', 'duration_ms', 'popularity', 'energy'], axis=1)
'''
  Description: Returns number of clusters, frequency and percentage for each
  given model prediction for user playlist

  -param[in] predictions: model prediction for user playlist
  -param[in] top_n: number of cluster classes

  -param[out] cluster_num: minimum between top_n and unique values
  -param[out] freq_clusters: clusters sorted by increasing freq
  -param[out] freq_percent: percentage that each cluster should be used in the recommendation
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
  -param[in] playlist: user playlist, list of track_id
  -param[in] song_data: dataset with all songs
  -param[in] scaler: fitted scaler
  -param[in] model: k-cluster model
  -param[in] rec_count_max: number of song recommendations
  -param[in] top_n: number of cluster classes

  -param[out] song_recs: song recommendations
'''
def get_recommendations(model, song_data, playlist, scaler, rec_count_max=5, top_n=3):
    # Transform and predict
    songs = playlist['track_id'].values.tolist()
    X = playlist.drop(['track_id'], axis=1)
    X = scaler.fit_transform(X)
    predictions = model.predict(X)

    data = get_relevant_cols(song_data)
    data = scaler.transform(data)

    song_recs = pd.DataFrame(columns=['track_id', 'similarity'])
    # Get most frequent cluster classes from user input
    cluster_num, freq_clusters, freq_perc = get_freq_clusters(predictions, top_n)
    for i in range(cluster_num):
        # Number of recommendations from given cluster
        rec_count = round(freq_perc[i] * rec_count_max)
        cur_cluster_number = freq_clusters[i]

        # Create mean vector and calculate similarity
        pos = np.where(predictions == cur_cluster_number)[0] 
        cluster_songs = X[pos, :]
        mean_song = np.mean(cluster_songs, axis=0)
        similarity = cdist(np.reshape(mean_song, (1,-1)), data)
        
        # Sort to get similar songs
        similarity_s = pd.Series(similarity.flatten(), name='similarity')
        similar_songs = pd.concat([song_data['track_id'].reset_index(drop=True), similarity_s.reset_index(drop=True)], axis=1)
        # Remove songs from user_songs list
        similar_songs = similar_songs[~(similar_songs['track_id'].isin(songs))]
        similar_songs = similar_songs.sort_values(by='similarity', ascending=True).reset_index(drop=True)
        song_recs = song_recs._append(similar_songs)
        
    song_recs = song_recs.reset_index(drop=True)
    return song_recs.loc[:rec_count_max-1]
