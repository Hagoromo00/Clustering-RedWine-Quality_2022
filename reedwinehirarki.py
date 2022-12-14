from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score

import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import time 
from datetime import date
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

wine = pd.read_csv('winequality-red.csv')
wine.shape
df = pd.read_csv('winequality-red.csv')
df.head()
print(df.head())
plt.figure(figsize = (20, 15))
df.columns
visual = sns.displot(df['pH'])
wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters =i, init = 'k-means++', random_state= 42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Methods Graphics')
plt.xlabel('Cluster')
plt.ylabel('WCSS')
plt.show()
model = KMeans()
visible = KElbowVisualizer(model, k=(1,10), timings = False)
visible.fit(df)
visible.show()


bca = PCA()
X = bca.fit_transform(df)
kmeans = KMeans(n_clusters=3)
label = kmeans.fit_predict(X)
graphics = np.unique(label)
for i in graphics:
    plt.scatter(X[label==i,0], X[label==i,1], label=i, s=20)
    
plt.legend()
plt.title('Wine After Drop pH')
plt.show()


#graphic
sil = []
for k in range(2, 11):
  kmeans = KMeans(n_clusters = k).fit(df)
  labels = kmeans.labels_
  sil.append(silhouette_score(df, labels, metric = 'euclidean'))

plt.plot(range(2,11), sil)
plt.xticks(range(2,11))
plt.xlabel("Number Clusters")
plt.ylabel("Coefficient Silhouette")
plt.show()

#score
for k in range(2,11):
    kmeans = KMeans(n_clusters=k).fit(df)
    Score = silhouette_score(df, kmeans.labels_, metric='euclidean')
    print("{} For Silhouette Score : {}".format(k,Score))