import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

url = 'https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv'

df_raw = pd.read_csv(url)

df = df_raw[['Latitude', 'Longitude', 'MedInc']]

df_scaled = StandardScaler().fit_transform(df)

kmeans = KMeans(n_clusters=2)

df["Cluster"] = kmeans.fit_predict(df_scaled)
df["Cluster"] = df["Cluster"].astype("category")

df.to_csv('../data/processed/clusters.csv')

filename = '../models/housting.pickle'
pickle.dump(kmeans, open(filename,'wb'))
