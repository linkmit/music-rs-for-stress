import pandas as pd
import numpy as np
import re
import seaborn as sns
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('audio_features.csv')

sns.histplot(df['key'])
plt.title("Frequency of keys present in the SMT Playlist")
plt.show()

sns.histplot(df['mode'])
plt.title("Frequency of modes in the SMT Playlist")
plt.show()

# pre-processing
def tempo_to_float(x):
    match = re.search(r"[-+]?\d*\.\d+|\d+", x)
    return float(match.group(0))
df['tempo'] = df['tempo'].apply(tempo_to_float)

# categorical values to numerical 
df = pd.get_dummies(df, columns = ['key','mode'])
# effectively, if mode_Major = False, it is in the Minor key
df.drop(columns='mode_Minor', inplace=True) 

# removing outlier
df = df[df['tempo'] != 0]

# scaling
X = df.iloc[:, 1:]
X_std = X.copy()
scaler = StandardScaler()
X_std.iloc[:, :7] = scaler.fit_transform(X.iloc[:, :7])
X_std = X_std.iloc[:,:7]

# some EDA to see correlations
correlation_matrix = X_std.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# flatness and contrast seem to no tbe giving many correlations
X_std.drop(columns=['bandwidth','rolloff'], inplace=True)
df.drop(columns=['bandwidth','rolloff'], inplace=True)

# as there are many features, we will do some dimensionality reduction
pca = PCA()
pca.fit(X_std)

# determining the best # of dimensions
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
plt.figure(figsize = (10,6))
plt.plot(range(1, len(per_var)+1), per_var.cumsum(), marker = "o", linestyle = "--")
plt.grid()
plt.ylabel("Percentage Cumulative of Explained Variance")
plt.xlabel("Number of Components")
plt.title("Explained Variance by Component")
plt.show()

# pca = PCA(n_components=5)
# X_pca = pca.fit_transform(X_std)

# elbow-method to obtain optimal k
WCSS = []
for i in range(1,10):
  kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
  kmeans.fit(X_std)
  WCSS.append(kmeans.inertia_)

plt.figure(figsize = (10,6))
plt.plot(range(1,10), WCSS, marker = "o", linestyle = "--")
plt.grid()
plt.title("Within-Cluster Sum of Squares vs. # Clusters")
plt.ylabel("WCSS")
plt.xlabel("# of Clusters")
plt.show()

# k-means clustering
kmeans= KMeans(n_clusters=4, init="k-means++", random_state=42)
kmeans.fit(X_std)

# if you want to plot a PCA-reduced 3D space of clustering results
# pca = PCA(n_components=3)
# X_pca = pca.fit_transform(X_std)
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans.labels_, cmap='viridis', marker='o')
# ax.set_title("KMeans Clustering (PCA-reduced 3D)")
# ax.set_xlabel("Principal Component 1")
# ax.set_ylabel("Principal Component 2")
# ax.set_zlabel("Principal Component 3")
# plt.show()

X_std['cluster'] = kmeans.labels_
df['cluster']= kmeans.labels_
cluster_means = X_std.groupby('cluster').median() 

print(X_std['cluster'].value_counts())

cluster_means.index = range(1, len(cluster_means) + 1)
sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Standardized median feature values by cluster')
plt.show()

cluster_1 = df[df['cluster']==0]
cluster_2 = df[df['cluster']==1]
cluster_3 = df[df['cluster']==2]
cluster_4 = df[df['cluster']==3]

cluster_1_median = cluster_1.iloc[:, 1:].median(axis=0)
cluster_2_median = cluster_2.iloc[:, 1:].median(axis=0)
cluster_3_median = cluster_3.iloc[:, 1:].median(axis=0)
cluster_4_median = cluster_4.iloc[:, 1:].median(axis=0)

medians_df = pd.DataFrame({
    'Cluster 1': cluster_1_median,
    'Cluster 2': cluster_2_median,
    'Cluster 3': cluster_3_median,
    'Cluster 4': cluster_4_median
})

medians_df = medians_df.T
medians_df.reset_index(inplace=True)
medians_df.rename(columns={'index': 'Cluster'}, inplace=True)

print(medians_df)
# medians_df.to_csv('cluster_medians.csv')

df_original = pd.read_csv('audio_features.csv')
cluster_1_keys = df_original['key'].iloc[cluster_1.index]
cluster_2_keys = df_original['key'].iloc[cluster_2.index]
cluster_3_keys = df_original['key'].iloc[cluster_3.index]
cluster_4_keys = df_original['key'].iloc[cluster_4.index]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of plots
axs[0, 0].hist(cluster_1_keys, bins=10, color='blue', alpha=0.7, edgecolor='black')
axs[0, 0].set_title('Cluster 1')

axs[0, 1].hist(cluster_2_keys, bins=10, color='green', alpha=0.7, edgecolor='black')
axs[0, 1].set_title('Cluster 2')

axs[1, 0].hist(cluster_3_keys, bins=10, color='red', alpha=0.7, edgecolor='black')
axs[1, 0].set_title('Cluster 3')

axs[1, 1].hist(cluster_4_keys, bins=10, color='purple', alpha=0.7,edgecolor='black')
axs[1, 1].set_title('Cluster 4')

for ax in axs.flat:
    ax.set_xlabel('Key')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()








