from turtle import distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
wine = load_wine()

data = wine.data
label = wine.target
columns = wine.feature_names

data = pd.DataFrame(data, columns=columns)
print(data.head())

# K-Means
# 데이터 전처리
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
print(data.shape) # 13차원의 데이터

#PCA (차원의 축소)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)

print(data) # 2차원 축소

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3) # 클러스터 갯수를 정해줘야함

kmeans.fit(data)

cluster = kmeans.predict(data)
print(type(cluster)) # numpy array

# print(cluster) # 분류 확인

plt.scatter(data[:,0], data[:,1],c=cluster, edgecolor= 'black', linewidths=1) #데이터의 0번 항목과 1번 항목(데이터는 현제 2차원), cluster 따라 컬러를 줌
#plt.show()

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
single_clustering = AgglomerativeClustering(n_clusters=3,linkage='single')
complete_clustering = AgglomerativeClustering(n_clusters=3,linkage='complete')
average_clustering = AgglomerativeClustering(n_clusters=3,linkage='average')

single_clustering.fit(data) # 가장 가까운 거리
complete_clustering.fit(data) # 가장 먼 거리
average_clustering.fit(data) # 평균값

single_cluster = single_clustering.labels_
complete_cluster = complete_clustering.labels_
average_cluster = average_clustering.labels_

print(single_cluster)
print(complete_cluster)
print(average_cluster)

plt.scatter(data[:,0], data[:,1], c=single_cluster)
#plt.show()
plt.scatter(data[:,0], data[:,1], c=complete_cluster)
#plt.show()
plt.scatter(data[:,0], data[:,1], c=average_cluster)
#plt.show()
plt.scatter(data[:,0], data[:,1], c=label)
#plt.show()

# dendrogram
from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(10,10))

children = single_clustering.children_
#print(children) # 어디 연결되었는지 보여줌
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
# 여기까지 준비과정

dendrogram(linkage_matrix, p=len(data), labels=single_cluster, show_contracted=True, no_labels=True)
#plt.show()

# Silhouette

from sklearn.metrics import silhouette_score

best_n = 1
best_score = -1

for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(data)
    cluster = kmeans.predict(data)
    score = silhouette_score(data, cluster)

    print('클러스터의 수: {} 실루엣 점수: {:.2f}'.format(n_cluster, score))

    if score > best_score:
        best_n = n_cluster
        best_score = score

print('가장 높은 실루엣 점수를 가진 클러스터 수: {}, 시루엣 점수: {:.2f}'.format(best_n, best_score))

# Silhouette

from sklearn.metrics import silhouette_score

best_n = 1
best_score = -1

for n_cluster in range(2,11):
  average_clustering = AgglomerativeClustering(n_clusters=n_cluster,linkage='average')
  average_clustering.fit(data)
  cluster = average_clustering.labels_
  score = silhouette_score(data, cluster)

  print('클러스터의 수: {} 실루엣 점수:{:.2f}'.format(n_cluster, score))

  if score > best_score:
    best_n = n_cluster
    best_score = score

print('가장 높은 실루엣 점수를 가진 클러스터 수: {}, 실루엣 점수 {:.2f}'.format(best_n, best_score))