# 10. Implementation of Hierarchical and DBSCAN Clustering

# Install required packages (run once in terminal):
# pip install numpy pandas matplotlib scipy scikit-learn

# ---------------------------------------------------------------
# Enter the Data Points
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

X = np.array([(5, 19), (4, 17), (11, 25), (6, 22), (12, 21),
              (10, 21), (14, 24), (3, 16), (10, 24), (4, 21)])

for i, p in enumerate(X):
    print("Point", i, "=", tuple(int(v) for v in p))

# ---------------------------------------------------------------
# Part A – Hierarchical (Agglomerative) Clustering
# ---------------------------------------------------------------

# Build the Merge Table
from scipy.cluster.hierarchy import linkage, dendrogram

Z = linkage(X, method="average")
print("merged distance size")
for a, b, dist, size in Z:
    print("%3d and %-3d %7.2f %d" % (a, b, dist, size))

# Draw the Dendrogram
plt.figure(figsize=(8, 4.5))
dendrogram(Z)
plt.title("Dendrogram (average linkage)")
plt.xlabel("Point number")
plt.ylabel("Distance")
plt.show()

# Form Two Clusters
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=2, linkage="average")
labels = model.fit_predict(X)

print("Cluster labels:", labels)
for c in [0, 1]:
    print("Cluster", c, ":", [tuple(int(v) for v in p)
                              for p in X[labels == c]])

# Plot the Clusters
plt.figure(figsize=(7, 4.5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow", s=90)
for i, p in enumerate(X):
    plt.annotate(str(i), (p[0] + 0.15, p[1] + 0.15))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Agglomerative Clustering (2 clusters)")
plt.grid(True)
plt.show()

# ---------------------------------------------------------------
# Part B – DBSCAN
# ---------------------------------------------------------------

# Create blob data
from sklearn.datasets import make_blobs

B, _ = make_blobs(n_samples=750, n_features=2, centers=3,
                  cluster_std=0.4, random_state=0)
print("Shape:", B.shape)
print(B[:3].round(2))

# Run DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=4).fit(B)
lab = dbscan.labels_

n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
print("Clusters found:", n_clusters)
print("Noise points:", int((lab == -1).sum()))
print("Cluster sizes:", np.bincount(lab[lab >= 0]))
print("Core points:", len(dbscan.core_sample_indices_))

# Plot the Clusters
plt.figure(figsize=(7, 4.5))
plt.scatter(B[lab >= 0, 0], B[lab >= 0, 1],
            c=lab[lab >= 0], cmap="viridis", s=15)
plt.scatter(B[lab == -1, 0], B[lab == -1, 1],
            c="black", marker="x", s=60, label="Noise")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("DBSCAN (eps = 0.5, min_samples = 4)")
plt.legend()
plt.show()

# Effect of eps
for e in [0.1, 0.2, 0.3, 0.5, 1.0]:
    l = DBSCAN(eps=e, min_samples=4).fit(B).labels_
    k = len(set(l)) - (1 if -1 in l else 0)
    print("eps =", e, " clusters =", k,
          " noise =", int((l == -1).sum()))

# ---------------------------------------------------------------
# K-Means vs DBSCAN on Non-Linear Data (Two Moons)
# ---------------------------------------------------------------
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

M, _ = make_moons(n_samples=300, noise=0.06, random_state=0)

km = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(M)
db = DBSCAN(eps=0.2, min_samples=4).fit_predict(M)

print("DBSCAN clusters:", len(set(db)) - (1 if -1 in db else 0),
      " noise:", int((db == -1).sum()))

# Compare the Two Results
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

ax[0].scatter(M[:, 0], M[:, 1], c=km, cmap="coolwarm", s=15)
ax[0].set_title("K-Means (K = 2)")

ax[1].scatter(M[:, 0], M[:, 1], c=db, cmap="coolwarm", s=15)
ax[1].set_title("DBSCAN (eps = 0.2)")

plt.show()

# K-means cuts each moon in two, because it can only form round groups.
# DBSCAN follows the dense curves and recovers the two moons.