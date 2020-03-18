import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import common

client = mlflow.tracking.MlflowClient()
print(f"MLflow Version: {mlflow.__version__}")

def train(X_train, X_test, y_train, y_test, max_depth):
    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow_version", mlflow.__version__)
        mlflow.log_param("max_depth", max_depth)
        dt = DecisionTreeRegressor(max_depth=max_depth)
        dt.fit(X_train, y_train)
        predictions = dt.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mlflow.log_metric("rmse", rmse)
        print(f"{rmse:5.3f} {max_depth:2d} {run.info.run_id} {run.info.experiment_id}")
        mlflow.sklearn.log_model(dt, "sklearn-model")

def run(experiment_name, data_path):
    print(f"==== {__file__} ====")
    mlflow.set_experiment(experiment_name)
    exp = client.get_experiment_by_name(experiment_name)
    print(f"Experiment ID: {exp.experiment_id}")

    # Delete existing runs
    runs = client.list_run_infos(exp.experiment_id)
    for run in runs:
        client.delete_run(run.run_id)

    # Train against different parameters
    X_train, X_test, y_train, y_test = common.build_data(data_path)
    params = (1, 2, 4, 16)
    print(f"Params: {params}")
    for p in params:
        train(X_train, X_test, y_train, y_test, p)

    # Find best run
    runs = client.search_runs(exp.experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
    best_run = runs[0]
    print(f"Best run: {best_run.data.metrics['rmse']:5.3f} {best_run.info.run_id}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default=common.experiment_name)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default=common.data_path)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    run(args.experiment_name, args.data_path)
