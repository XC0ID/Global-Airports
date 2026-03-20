"""
main.py
-------
Entry point for the Global Airports ML project.
Runs the full pipeline: preprocessing → clustering → classification.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing  import run_preprocessing_pipeline, scale_features, CLUSTER_FEATURES
from src.clustering     import kmeans_clustering, run_clustering_pipeline
from src.classification import run_classification_pipeline


def main():
    print("=" * 55)
    print("   GLOBAL AIRPORTS — ML PIPELINE")
    print("=" * 55)

    print("\n[STEP 1] Preprocessing...")
    df = run_preprocessing_pipeline(filepath="data/airports.csv")

    print("\n[STEP 2] Clustering airports...")
    X, scaler = scale_features(df, feature_cols=CLUSTER_FEATURES)
    df_clustered, cluster_model = run_clustering_pipeline(X, df, n_clusters=4)

    print("\nSample clustered airports:")
    print(df_clustered[["name", "country", "passengers_millions", "cluster"]]
          .sort_values("cluster")
          .to_string(index=False))

    print("\n[STEP 3] Training classifier...")
    results = run_classification_pipeline(df)

    print("\n" + "=" * 55)
    print("   PIPELINE COMPLETE")
    print(f"   Final Test Accuracy : {results['metrics']['accuracy']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
