import pandas as pd
import numpy as np
from tabulate import tabulate

def read_prediction_data(data_path):
    data = pd.read_csv(data_path)
    if 'quality' in data:
         data = data.drop(['quality'], axis=1)
    print("data.shape:",data.shape)
    return data

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

