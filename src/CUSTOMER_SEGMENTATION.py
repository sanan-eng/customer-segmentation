import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Mall_Customers.csv')
print(df)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df[['Annual Income (k$)','Spending Score (1-100)']].describe())
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_scaled=scalar.fit_transform(df[['Annual Income (k$)','Spending Score (1-100)']])
from sklearn.cluster import KMeans
inertia=[]
K=range(1,11)
for k in K:
    kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K,inertia,'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Ineertia(Within-cluster sum of squares')
plt.title('Elbow method for optimal K')
plt.show()
kmeans=KMeans(n_clusters=5,random_state=42,n_init=10)
y_kmeans=kmeans.fit_predict(X_scaled)
df['Cluster']=y_kmeans
print(df.head())
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y_kmeans,cmap='viridis',s=50)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='red',marker='X',label='Centroids')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.title('Customer Segments (k=5)')
plt.legend()
plt.show()
from sklearn.neighbors import NearestNeighbors
neigh=NearestNeighbors(n_neighbors=5)
nbrs=neigh.fit(X_scaled)
distances,indices=nbrs.kneighbors(X_scaled)
distances=np.sort(distances[:,4])
plt.plot(distances)
plt.xlabel('points sorted by distance')
plt.ylabel('5th nearest neighbor distance')
plt.title('k-distance graph for choosing eps')
plt.show()
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.5,min_samples=5)
labels=dbscan.fit_predict(X_scaled)
df['Cluster_DBSCAN']=labels
print(df)
print("Noise points in DBSCAN :",(labels== -1).sum())
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=labels,cmap='plasma')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.title('Customer Segments using DBSCAN')
plt.show()
avg_spending_kmeans=df.groupby('Cluster')['Spending Score (1-100)'].mean()
print('Average spending per cluster (KMEANS):')
print(avg_spending_kmeans)
avg_spending_dbscan=df[df['Cluster_DBSCAN']!=-1].groupby('Cluster_DBSCAN')['Spending Score (1-100)'].mean()
print('Average spending per cluster (DBSCAN):')
print(avg_spending_dbscan)