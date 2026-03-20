"""
clustering.py
-------------
KMeans and Hierarchical clustering for Global Airports.
Includes elbow method, silhouette analysis, and cluster labelling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def plot_elbow(X: np.ndarray,
               k_range: range = range(2, 11),
               save_path: str = None) -> None:
    """
    Plot the Within-Cluster Sum of Squares (WCSS) elbow curve
    to help choose the optimal number of clusters.
    """
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), wcss, marker="o", color="steelblue", linewidth=2)
    plt.title("Elbow Method — Optimal K", fontsize=14)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Inertia)")
    plt.xticks(list(k_range))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[✔] Elbow plot complete.")


def plot_silhouette(X: np.ndarray,
                    k_range: range = range(2, 8),
                    save_path: str = None) -> int:
    """
    Plot average silhouette score for each K and return the best K.
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"  K={k}  →  Silhouette Score = {score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\n[✔] Best K by silhouette: {best_k}  (score = {scores[best_k]:.4f})")

    plt.figure(figsize=(8, 4))
    plt.bar(list(scores.keys()), list(scores.values()), color="steelblue", alpha=0.8)
    plt.axvline(x=best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    plt.title("Silhouette Score per K", fontsize=14)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return best_k


def kmeans_clustering(X: np.ndarray,
                      n_clusters: int = 4,
                      random_state: int = 42) -> tuple:
    """
    Fit KMeans and return (labels, model, silhouette_score).

    Parameters
    ----------
    X           : Scaled feature matrix (np.ndarray)
    n_clusters  : Number of clusters
    random_state: Reproducibility seed

    Returns
    -------
    labels      : np.ndarray  — cluster label per airport
    model       : fitted KMeans object
    score       : float       — silhouette score
    """
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    score  = silhouette_score(X, labels)

    print(f"[✔] KMeans fitted  |  K={n_clusters}  |  Silhouette={score:.4f}  |  Inertia={model.inertia_:.2f}")
    return labels, model, score


def hierarchical_clustering(X: np.ndarray,
                             n_clusters: int = 4,
                             linkage: str = "ward") -> tuple:
    """
    Fit Agglomerative (Hierarchical) Clustering.

    Returns
    -------
    labels  : np.ndarray
    model   : fitted AgglomerativeClustering
    score   : silhouette score
    """
    model  = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    score  = silhouette_score(X, labels)

    print(f"[✔] Hierarchical clustering done  |  K={n_clusters}  |  Silhouette={score:.4f}")
    return labels, model, score


# ──────────────────────────────────────────────
# 5. CLUSTER VISUALISATION (PCA 2D)
# ──────────────────────────────────────────────

def plot_clusters_pca(X: np.ndarray,
                      labels: np.ndarray,
                      title: str = "Airport Clusters (PCA 2D)",
                      save_path: str = None) -> None:
    """Reduce to 2D with PCA and scatter-plot the clusters."""
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X)

    n_clusters = len(np.unique(labels))
    colors     = cm.tab10(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(9, 6))
    for idx, (k, color) in enumerate(zip(np.unique(labels), colors)):
        mask = labels == k
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[color], label=f"Cluster {k}",
                    s=80, alpha=0.8, edgecolors="black", linewidth=0.4)

    plt.title(title, fontsize=14)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend(title="Cluster")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def cluster_profile(df: pd.DataFrame,
                    labels: np.ndarray,
                    feature_cols: list = None) -> pd.DataFrame:
    """
    Attach cluster labels to the dataframe and show
    mean statistics per cluster.
    """
    df = df.copy()
    df["cluster"] = labels

    if feature_cols is None:
        feature_cols = ["passengers_millions", "runways",
                        "altitude_ft", "latitude", "longitude"]

    profile = df.groupby("cluster")[feature_cols].mean().round(2)
    profile["count"] = df.groupby("cluster").size()

    print("\n══════════ CLUSTER PROFILES ══════════")
    print(profile.to_string())
    return profile


def plot_clusters_geo(df: pd.DataFrame,
                      labels: np.ndarray,
                      save_path: str = None) -> None:
    """
    Plot airport clusters on a world scatter map
    using latitude / longitude.
    """
    df = df.copy()
    df["cluster"] = labels
    n_clusters = len(np.unique(labels))
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(14, 7))

    ax = plt.gca()
    ax.set_facecolor("#e8f4f8")

    for k, color in zip(np.unique(labels), colors):
        mask = df["cluster"] == k
        plt.scatter(
            df.loc[mask, "longitude"],
            df.loc[mask, "latitude"],
            c=[color], label=f"Cluster {k}",
            s=100, alpha=0.85, edgecolors="black", linewidth=0.5, zorder=5
        )

    plt.title("Global Airport Clusters — Geographic View", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.legend(title="Cluster", loc="lower left")
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_clustering_pipeline(X: np.ndarray,
                             df: pd.DataFrame,
                             n_clusters: int = 4) -> pd.DataFrame:
    """
    Run the complete clustering pipeline and return
    the dataframe with a 'cluster' column added.
    """
    print("\n══════════ CLUSTERING PIPELINE ══════════")

    # Elbow & silhouette for guidance
    plot_elbow(X)
    best_k = plot_silhouette(X)

    # Use provided n_clusters (or best_k if you prefer)
    labels, model, score = kmeans_clustering(X, n_clusters=n_clusters)

    # Visualise
    plot_clusters_pca(X, labels)
    plot_clusters_geo(df, labels)

    # Profile
    profile = cluster_profile(df, labels)

    df = df.copy()
    df["cluster"] = labels
    print("\n[✔] Clustering pipeline complete.")
    return df, model


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline, scale_features, CLUSTER_FEATURES
    df = run_preprocessing_pipeline()
    X, scaler = scale_features(df, CLUSTER_FEATURES)
    df_clustered, model = run_clustering_pipeline(X, df, n_clusters=4)
    print(df_clustered[["name", "country", "passengers_millions", "cluster"]].to_string())
