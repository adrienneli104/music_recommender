# Music Recommender System | Adrienne Li

I built a music recommendation system where users can input a Spotify playlist link and receive song recommendations based on the songs in that playlist. 
They can specify how many song recommendations they want and the suggestions are chosen based on similar audio features, such as tempo, energy, and loudness.  
## User Guide
```
git clone https://github.com/adrienneli104/music_recommender.git
pip3 install -r requirements.txt
python wsgi.py
```
View on http://127.0.0.1:5000/
## Resources Used
The [dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/) I used was found on Kaggle and it stores Spotify songs with their audio features.  <br>

I referenced Eric Chang's [web app code](https://github.com/enjuichang/PracticalDataScience-ENCA/tree/main) as a starting point to build the basics of my application such
as connecting the form view to the backend. However, I used a different data set, created my own model with a different algorithm, implemented Spotify iFrames, 
the visualizations and Javascript associated to make it interactive. 

## Model Algorithm
I used the K-Means Clustering Algorithm for the recommendation model which groups songs together with similar features into clusters so that when given one song, 
we can find songs in the same cluster that shares similar features. K-Means is an unsupervised machine learning algorithm that groups the data into clusters. 
The goal is to group the data such that data in each subgroup is similar to each other but the subgroups among themselves are different from each other. 
As the name suggests, the algorithm creates "k" clusters, finds the cluster center, and groups the data such that the sum of squared distances between each data 
point and the nearest center is minimized. Sum-of-squared error (SSE) measures how close the data points are to their cluster center, which we will use to 
determine how well the data fits into their assigned subgroups. <br>

I used the elbow method to determine how many k clusters there should be. This calculates the Within-Cluster-Sum of Squared Errors (WCSS) for different values of k.
I chose the value of k right as WCSS begins to decrease, which looks like an elbow when plotted. 
