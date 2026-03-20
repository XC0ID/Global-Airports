import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    drop_cols = ['id', 'ident', 'name']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    df = df.dropna()
    return df


def encode_features(df):
    if 'type' in df.columns:
        le = LabelEncoder()
        df['type'] = le.fit_transform(df['type'])

    if 'scheduled_service' in df.columns:
        df['scheduled_service'] = df['scheduled_service'].map({'yes': 1, 'no': 0})

    return df


def scale_features(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def preprocess_pipeline(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    return df