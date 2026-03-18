from sklearn.cluster import KMeans

def apply_kmeans(data, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels