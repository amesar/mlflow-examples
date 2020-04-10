import pandas as pd

def read_prediction_data(data_path):
    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_json(data_path)
    if 'quality' in df:
         df = df.drop(['quality'], axis=1)
    return df
