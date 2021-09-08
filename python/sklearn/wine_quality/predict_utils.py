import pandas as pd
import numpy as np
from tabulate import tabulate
import mlflow

print("MLflow Version:", mlflow.__version__)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

def read_prediction_data(data_path):
    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_json(data_path)
    if "quality" in df:
        df = df.drop(["quality"], axis=1)
    print("Data:")
    print("  shape:",df.shape)
    print("  dtypes:")
    for x in zip(df.columns,df.dtypes):
        print(f"    {x[0]}: {x[1]}")
    return df

def display_predictions(predictions):
    print("predictions.type:",type(predictions))
    print("predictions.shape:",predictions.shape)
    if isinstance(predictions,np.ndarray):
        predictions = np.round(predictions, decimals=3)
        predictions = pd.DataFrame(predictions,columns=["prediction"])
    else:
        predictions = predictions.round(3)
    df = predictions.head(5)
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
