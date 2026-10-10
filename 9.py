# 9. Implementation of K-Means
# Enter the Data Points
# The points are stored in a NumPy array with one row per point, and saved to a CSV file.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X = np.array([(5,19), (4,17), (11,25), (6,22), (12,21),
(10,21), (14,24), (3,16), (10,24), (4,21)])
pd.DataFrame(X, columns=["x", "y"]).to_csv("points.csv", index=False)
print("Shape:", X.shape)
# Plot the Raw Points
plt.figure(figsize=(7, 4.5))
plt.scatter(X[:, 0], X[:, 1], s=80, color="gray")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Data Points")
plt.grid(True)
plt.show()
# Elbow Method: WCSS for K = 1 to 6
# Within-Cluster Sum of Squares (WCSS)
# K-means is run for each K and its WCSS (inertia_) is stored. n_init=10 tries 10 different starting
# centroids and keeps the best.
# WCSS measures how far data points are from the centroid of their assigned cluster.
from sklearn.cluster import KMeans
wcss = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)
    print("K =", k, " WCSS =", round(km.inertia_, 2))
# Plot the Elbow Curve
# WCSS always falls as K grows. The best K is at the elbow: the point after which the fall becomes
# small.
plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 7), wcss, color="red", marker="8")
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of clusters K"); plt.ylabel("WCSS")
plt.grid(True)
plt.show()
# Run K-Means with K = 2
model = KMeans(n_clusters=2, random_state=0, n_init=10)
y_means = model.fit_predict(X)
print("Cluster labels:", y_means)
print("Centroids:")
print(model.cluster_centers_)
print("WCSS:", round(model.inertia_, 2))
print("Iterations:", model.n_iter_)
# Plot the Clusters and Centroids
plt.figure(figsize=(7, 4.5))
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=80,
c="blue", label="Cluster 0")
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=80,
c="orange", label="Cluster 1")
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
s=160, marker="s", c="red", label="Centroids")
plt.xlabel("x"); plt.ylabel("y"); plt.title("K-Means Clusters (K = 2)")
plt.legend(); plt.grid(True)
plt.show()
# Silhouette Score
from sklearn.metrics import silhouette_score
print("Silhouette for K = 2:",
round(silhouette_score(X, y_means), 3))
for k in range(2, 6):
    lab = KMeans(n_clusters=k, random_state=0,
    n_init=10).fit_predict(X)
    print("K =", k, " silhouette =",
    round(silhouette_score(X, lab), 3))
# Homogeneity and Completeness
from sklearn.metrics import homogeneity_score, completeness_score
y_true = [0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
print("Homogeneity :", homogeneity_score(y_true, y_means))
print("Completeness:", completeness_score(y_true, y_means))
lab3 = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X)
print("K = 3 homogeneity :", round(homogeneity_score(y_true, lab3), 3))
print("K = 3 completeness:", round(completeness_score(y_true, lab3), 3))
# Assign a New Point
new = np.array([[7, 20]])
c = model.predict(new)[0]
print("Point (7, 20) belongs to cluster", c)
d0 = np.linalg.norm(new - model.cluster_centers_, axis=1)
print("Distances to the centroids:", d0.round(2))