from src.preprocessing import preprocess_pipeline
from src.clustering import kmeans_clustering
from src.classification import train_model, save_model

def main():
    path = "data/airports.csv"

    df = preprocess_pipeline(path)

    model = train_model(df)
    save_model(model, "models/airport_classifier.pkl")

    df, _ = kmeans_clustering(df, 3)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()