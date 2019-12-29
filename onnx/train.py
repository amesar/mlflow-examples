from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import mlflow
import mlflow.sklearn
import mlflow.onnx
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
client = mlflow.tracking.MlflowClient()

colLabel = "quality"

def build_data(data_path):
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.30, random_state=2019)
    X_train = train.drop([colLabel], axis=1)
    X_test = test.drop([colLabel], axis=1)
    y_train = train[[colLabel]]
    y_test = test[[colLabel]]
    return X_train, X_test, y_train, y_test 

def train(data_path, max_depth, max_leaf_nodes):
    X_train, X_test, y_train, y_test = build_data(data_path)
    with mlflow.start_run() as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print("MLflow:")
        print("  run_id:", run_id)
        print("  experiment_id:", experiment_id)
        print("  experiment_name:", client.get_experiment(experiment_id).name)

        # Create model
        dt = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        print("Model:\n ", dt)

        # Fit and predict
        dt.fit(X_train, y_train)
        predictions = dt.predict(X_test)

        # MLflow params
        print("Parameters:")
        print("  max_depth:", max_depth)
        print("  max_leaf_nodes:", max_leaf_nodes)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

        # MLflow metrics
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
        
        # MLflow tags
        mlflow.set_tag("data_path", data_path)
        mlflow.set_tag("mlflow_version", mlflow.version.VERSION)
        mlflow.set_tag("onnx_version", onnx.__version__)

        # MLflow log skearn model
        mlflow.sklearn.log_model(dt, "sklearn-model")

        # Convert sklearn model to ONNX and log model
        initial_type = [('float_input', FloatTensorType([None, X_test.shape[0]]))]
        onnx_model = convert_sklearn(dt, initial_types=initial_type)
        print("onnx_model.type:",type(onnx_model))
        mlflow.onnx.log_model(onnx_model, "onnx-model")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", required=True)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../data/wine-quality-white.csv")
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=None, type=int)
    parser.add_argument("--max_leaf_nodes", dest="max_leaf_nodes", help="max_leaf_nodes", default=None, type=int)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    mlflow.set_experiment(args.experiment_name)
    train(args.data_path, args.max_depth, args.max_leaf_nodes)
