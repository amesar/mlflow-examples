from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow
import mlflow.xgboost

print("Tracking URI:", mlflow.tracking.get_tracking_uri())
print("MLflow Version:", mlflow.version.VERSION)
print("XGBoost version:",xgb.__version__)

def build_data(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(["quality"], axis=1)
    y = data["quality"]
    return X, y

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="model_uri", required=True)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../data/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        
    model = mlflow.xgboost.load_model(args.model_uri)
    print("model:", model)
    
    X,y  = build_data(args.data_path)
    X = xgb.DMatrix(X, label=y)
    predictions = model.predict(X)
    print("predictions:", predictions)
