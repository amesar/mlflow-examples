import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import mlflow.onnx
import catboost
from catboost import CatBoostRegressor

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
#print("XGBoost version:",xgb.__version__)
print("Catboost version:",catboost.__version__)
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

def train(data_path, iterations, learning_rate, depth, log_as_onnx):
    X_train, X_test, y_train, _ = build_data(data_path)
    with mlflow.start_run() as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print("MLflow:")
        print("  run_id:", run_id)
        print("  experiment_id:", experiment_id)
        print("  experiment_name:", client.get_experiment(experiment_id).name)

        # MLflow params
        print("Parameters:")
        print("  depth:", depth)
        print("  learning_rate:", learning_rate)
        print("  iterations:", iterations)
        mlflow.log_param("depth", depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("iterations", iterations)

        # Create and fit model
        model = CatBoostRegressor(iterations=iterations,
                          learning_rate=learning_rate,
                          depth=depth)
        model.fit(X_train, y_train, verbose=False)
        print("model.type=",type(model))

        predictions = model.predict(X_test)
        print("Predictions:",predictions)

        # Log catboost model
        mlflow.sklearn.log_model(model, "catboost-model")

        # Log ONNX model
        if log_as_onnx:
            path = "catboost.onnx"
            model.save_model(path, format="onnx")
            with open(path, "rb") as f:
                onnx_model = f.read()
            mlflow.onnx.log_model(onnx_model, "onnx-model")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment name", required=True)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default="../../data/train/wine-quality-white.csv")
    parser.add_argument("--iterations", dest="iterations", help="Iterations", default=2, type=int)
    parser.add_argument("--depth", dest="depth", help="Depth", default=2, type=int)
    parser.add_argument("--learning_rate", dest="learning_rate", help="Learning rate", default=1, type=int)
    parser.add_argument("--log_as_onnx", dest="log_as_onnx", help="Log model as ONNX flavor", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    mlflow.set_experiment(args.experiment_name)
    train(args.data_path, args.iterations, args.learning_rate, args.depth, args.log_as_onnx)
