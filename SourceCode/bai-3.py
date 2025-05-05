import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns


CSV_FILE = 'results.csv'  
ENCODING = 'utf-8-sig'
OPTIMAL_K = 3  

#Đọc dữ liệu 
try:
    df = pd.read_csv(CSV_FILE, encoding=ENCODING)
    print(f"Reading file '{CSV_FILE}' successfully.")
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

#Lấy cột số
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_columns:
    print("No columns to analyze.")
    exit()

#Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_columns])

#Elbow Method để tìm số cụm
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

elbow_k = 3
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow at k={elbow_k}')
plt.annotate('⬅ The most obvious Elbow', xy=(elbow_k, inertia[elbow_k-1]), 
             xytext=(elbow_k + 3, inertia[elbow_k-1] + 100),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.title("Elbow chart - Select the optimal number of clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Phân cụm với KMeans
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

#Tóm tắt thống kê theo cụm 
print("\nAverage of cluster indices:")
cluster_summary = df.groupby('Cluster')[numeric_columns].mean().round(2)
print(cluster_summary)

#Giảm chiều bằng PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

#Biểu đồ phân cụm 2D 
df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']

plt.figure(figsize=(10, 8))
palette = {0: 'green', 1: 'blue', 2: 'red'}
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca,
                palette = palette, s=100, alpha=0.7)
plt.title("Player clustering (PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="CLuster")
plt.grid(True)
plt.tight_layout()
plt.show()


print("Comment:")
print("Choosing the number of clusters k=3 in the KMeans method is reasonable because, based on the Elbow Method, we observe a clear 'elbow' point at k=3. When plotting the 'inertia' (the total distance between data points and their cluster center), we notice that after k=3, the decrease in inertia becomes less significant. This indicates that three clusters provide the best division of the data, as increasing the number of clusters beyond k=3 does not substantially improve the dispersion of the points.")
print("As we can see in the Elbow chart, k = 2 can be a good way to cluster but there are some reasons for not doing that like ignoring an important division in the data, resulting in a lack of clear distinction between different groups of players if k = 2 is chosen.")
print("- Cluster 0: Defensive player(defender , midfielder, goalkeeper , etc..) (more in-time game, less attacking).")
print("- Cluster 1: Substitute player, little contribution.")
print("- Cluster 2: Attacking player with high scores/assists.")
