import random
import numpy as np
import pandas as pd
import sys
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler

def get_data(data):
    """
    Load and preprocess data.
    
    """
    df = pd.read_csv(data, header=None)
    label_mapping = {chr(97 + i): i + 1 for i in range(26)}
    df.iloc[:, -1] = df.iloc[:, -1].map(label_mapping)
    df.iloc[:, :-1] = MinMaxScaler().fit_transform(df.iloc[:, :-1])
    print("Data sample:\n", df.head())
    return df.values.tolist()

def is_converged(centroids, old_centroids):
    """
    Check if centroids have converged.
    
    """
    return set(map(tuple, centroids)) == set(map(tuple, old_centroids))

def get_distance(x, c):
    """
    Compute Euclidean distance.
    
    """
    return np.linalg.norm(np.array(x) - np.array(c))

def get_clusters(X, centroids):
    """
    Assign points to the nearest centroid.
    
    """
    clusters = defaultdict(list)
    for x in X:
        cluster = np.argmin([get_distance(x[:-1], c[:-1]) for c in centroids])
        clusters[cluster].append(x)
    return clusters

def get_centroids(old_centroids, clusters):
    """
    Compute new centroids as mean of clusters.
    
    """
    new_centroids = []
    for k in sorted(clusters.keys()):
        if clusters[k]:
            new_centroid = np.mean(clusters[k], axis=0)
            new_centroid[-1] = Counter(np.array(clusters[k])[:, -1]).most_common(1)[0][0]
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(old_centroids[k])
    return new_centroids

def find_centers(X, K):
    """
    Run K-means clustering until convergence.
    
    """
    old_centroids = random.sample(X, K)
    centroids = random.sample(X, K)
    iteration = 0
    while not is_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = get_clusters(X, centroids)
        centroids = get_centroids(old_centroids, clusters)
        iteration += 1
    return centroids, clusters, iteration

def get_purity(clusters, num_instances):
    """
    Compute purity score.
    
    """
    return sum(Counter(np.array(clusters[k])[:, -1]).most_common(1)[0][1] for k in clusters) / num_instances

data, k, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
X = get_data(data)
num_instances = len(X)

best_score, best_centroids, best_clusters, best_iteration = 0, [], [], 0
for _ in range(5):
    centroids, clusters, iteration = find_centers(X, k)
    purity = get_purity(clusters, num_instances)
    if purity > best_score:
        best_score, best_centroids, best_clusters, best_iteration = purity, centroids, clusters, iteration

label_mapping = {i + 1: chr(97 + i) for i in range(26)}
for c in best_centroids:
    c[-1] = label_mapping[c[-1]]

print(f"Best purity score: {best_score:.6f}\nIterations: {best_iteration}")
with open(output, 'w') as out:
    for k in best_clusters:
        out.write(f"Centroid {k}: {best_centroids[k]}\nPoints:\n")
        out.writelines(f"{pt}\n" for pt in best_clusters[k])
        out.write('\n' * 3)


import random
import numpy as np
from collections import defaultdict, Counter, deque

def get_data(filepath):
    """
    Load and preprocess categorical data from file.
    
    """
    with open(filepath, 'r') as f:
        instances = [list(deque(line.strip().split('\t')).rotate(-1) or line) for line in f]
    return instances

def is_converged(centroids, old_centroids):
    """
    Check if centroids have converged.
    
    """
    return set(map(tuple, centroids)) == set(map(tuple, old_centroids))

def get_distance(x, c):
    """
    Compute Hamming distance for categorical attributes.
    
    """
    return np.sum(np.array(x) != np.array(c), axis=0)

def get_clusters(X, centroids):
    """
    Assign each data point to the closest centroid.
    
    """
    clusters = defaultdict(list)
    for x in X:
        cluster = np.argmin([get_distance(x[:-1], c[:-1]) for c in centroids])
        clusters[cluster].append(x)
    return clusters

def get_centroids(clusters, num_features):
    """
    Compute new centroids using mode of each cluster.
    
    """
    new_centroids = []
    for k in sorted(clusters.keys()):
        points = np.array(clusters[k])
        mode = [Counter(points[:, i]).most_common(1)[0][0] for i in range(num_features)]
        mode.append('PAD')  # Placeholder for class label
        new_centroids.append(mode)
    return new_centroids

def find_centers(X, K):
    """
    Run K-modes clustering until convergence.
    
    """
    old_centroids = random.sample(X, K)
    centroids = random.sample(X, K)
    iteration = 0
    while not is_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = get_clusters(X, centroids)
        centroids = get_centroids(clusters, len(X[0]) - 1)
        iteration += 1
    return centroids, clusters, iteration

def get_purity(clusters, num_instances):
    """
    Compute purity score.
    
    """
    return sum(Counter(np.array(clusters[k])[:, -1]).most_common(1)[0][1] for k in clusters) / num_instances

# Load data and run clustering
DATA_PATH = 'clustering.training'
K = 2
X = get_data(DATA_PATH)
num_instances = len(X)
centroids, clusters, iteration = find_centers(X, K)
purity = get_purity(clusters, num_instances)

# Print class distributions in clusters
for k, points in clusters.items():
    class_attr = Counter(np.array(points)[:, -1]).most_common(1)
    print(class_attr)

print(f'\nThe purity for the task is {purity:.6f}')
