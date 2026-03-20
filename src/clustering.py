from sklearn.cluster import KMeans

def run_kmeans(data, n_clusters=5, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(data)
    return labels, model