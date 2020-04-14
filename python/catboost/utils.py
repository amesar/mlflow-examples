
import pandas as pd

def build_data(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(["quality"], axis=1)
    y = data["quality"]
    return X, y
