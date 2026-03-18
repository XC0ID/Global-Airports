import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def feature_engineering(df):
    df['lat_lon_sum'] = df['latitude'] + df['longitude']
    return df

def scale_features(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler