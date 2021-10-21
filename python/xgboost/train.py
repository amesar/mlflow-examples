import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow
import mlflow.xgboost

print("MLflow Version:", mlflow.__version__)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("XGBoost version:",xgb.__version__)
client = mlflow.tracking.MlflowClient()

def build_data(data_path):
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.30, random_state=2019)

    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)
    y_train = train["quality"]
    y_test = test["quality"]

    return X_train, X_test, y_train, y_test 

def train(data_path, max_depth, min_child_weight, estimators, model_name):
    X_train, X_test, y_train, y_test = build_data(data_path)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        print("MLflow:")
        print("  run_id:", run_id)
        print("  experiment_id:", experiment_id)
        print("  experiment_name:", client.get_experiment(experiment_id).name)

        # MLflow params
        print("Parameters:")
        print("  max_depth:", max_depth)
        print("  min_child_weight:", min_child_weight)
        print("  estimators:", estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_child_weight", min_child_weight)
        mlflow.log_param("estimators", estimators)

        # Create and fit model
        model = xgb.XGBRegressor(
                 max_depth=max_depth,
                 min_child_weight=min_child_weight,
                 random_state=42) 
        model.fit(X_train, y_train)

        # MLflow metrics
        predictions = model.predict(X_test)
        print("predictions:",predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print("Metrics:")
        print("  rmse:", rmse)
        print("  mae:", mae)
        print("  r2:", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log model
        mlflow.xgboost.log_model(model, "xgboost-model", registered_model_name=model_name)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment name", default=None)
    parser.add_argument("--model_name", dest="model_name", help="Registered model name", default=None)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default="../../data/train/wine-quality-white.csv")
    parser.add_argument("--estimators", dest="estimators", help="Estimators", default=10, type=int)
    parser.add_argument("--max_depth", dest="max_depth", help="Max depth", default=3, type=int)
    parser.add_argument("--min_child_weight", dest="min_child_weight", help="Min child weight", default=1.5, type=float)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    model_name = None if not args.model_name or args.model_name == "None" else args.model_name
    train(args.data_path, args.max_depth, args.min_child_weight, args.estimators, model_name)
