"""
preprocessing.py
----------------
Handles all data loading, cleaning, and feature preparation
for the Global Airports ML project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import os


def load_data(filepath: str = "data/airports.csv") -> pd.DataFrame:
    """Load the airports CSV into a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[✔] Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset."""
    print("\n══════════ DATA OVERVIEW ══════════")
    print(f"Shape       : {df.shape}")
    print(f"Columns     : {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:")
    print(df.describe())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw airport dataframe:
      - Drop duplicates
      - Fill / drop missing values
      - Fix data types
    """
    df = df.copy()

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[✔] Dropped {before - len(df)} duplicate rows.")

    df.dropna(subset=["latitude", "longitude", "passengers_millions"], inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna("Unknown", inplace=True)

    df["altitude_ft"] = pd.to_numeric(df["altitude_ft"], errors="coerce").fillna(0)
    df["runways"]     = pd.to_numeric(df["runways"],     errors="coerce").fillna(1).astype(int)

    print(f"[✔] Data cleaned. Final shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.
    """
    df = df.copy()

    df["altitude_category"] = pd.cut(
        df["altitude_ft"],
        bins=[-1, 500, 2000, 5000, 99999],
        labels=["Low", "Medium", "High", "Very High"]
    )

    df["passenger_tier"] = pd.cut(
        df["passengers_millions"],
        bins=[0, 20, 50, 80, 9999],
        labels=["Small", "Medium", "Large", "Mega"]
    )

    df["northern_hemisphere"] = (df["latitude"] >= 0).astype(int)
    df["eastern_hemisphere"]  = (df["longitude"] >= 0).astype(int)

    df["passengers_per_runway"] = (
        df["passengers_millions"] / df["runways"].replace(0, 1)
    ).round(2)

    def extract_region(tz: str) -> str:
        tz = str(tz)
        if "America" in tz:   return "Americas"
        if "Europe" in tz:    return "Europe"
        if "Asia" in tz:      return "Asia"
        if "Africa" in tz:    return "Africa"
        if "Australia" in tz: return "Oceania"
        return "Other"

    df["region"] = df["timezone"].apply(extract_region)

    print(f"[✔] Feature engineering done. New shape: {df.shape}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object / category columns."""
    df = df.copy()
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    print(f"[✔] Encoded {len(cat_cols)} categorical columns.")
    return df


CLUSTER_FEATURES = [
    "latitude", "longitude", "altitude_ft",
    "passengers_millions", "runways", "passengers_per_runway"
]

CLASSIFICATION_FEATURES = [
    "latitude", "longitude", "altitude_ft",
    "passengers_millions", "runways", "passengers_per_runway",
    "northern_hemisphere", "eastern_hemisphere"
]


def scale_features(df: pd.DataFrame,
                   feature_cols: list = None,
                   method: str = "standard") -> tuple:
    """
    Scale numeric features.

    Returns
    -------
    X_scaled  : np.ndarray
    scaler    : fitted scaler object
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    X = df[feature_cols].copy().fillna(0)

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[✔] Scaled {len(feature_cols)} features using {method} scaler.")
    return X_scaled, scaler


def prepare_classification_data(df: pd.DataFrame) -> tuple:
    """
    Prepare X and y for the classification task.
    Target: hub_type  (International / Regional / Domestic)
    """
    from sklearn.model_selection import train_test_split

    le = LabelEncoder()
    y = le.fit_transform(df["hub_type"].astype(str))
    X = df[CLASSIFICATION_FEATURES].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"[✔] Train size: {X_train_s.shape}  |  Test size: {X_test_s.shape}")
    return X_train_s, X_test_s, y_train, y_test, scaler, le


def run_preprocessing_pipeline(filepath: str = "data/airports.csv") -> pd.DataFrame:
    """End-to-end preprocessing pipeline."""
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    print("\n[✔] Preprocessing pipeline complete.")
    return df


if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    inspect_data(df)
