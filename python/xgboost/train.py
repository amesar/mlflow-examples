import click
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow
from common import opt_data_path


print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("MLflow Version:", mlflow.__version__)
print("XGBoost version:", xgb.__version__)
client = mlflow.tracking.MlflowClient()


def build_data(data_path):
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.30, random_state=2019)
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
        mlflow.set_tag("version.mlflow",mlflow.__version__)
        mlflow.set_tag("version.xgboost",xgb.__version__)

        # Create and fit model
        model = xgb.XGBRegressor(
                 max_depth = max_depth,
                 min_child_weight = min_child_weight,
                 n_estimators = estimators,
                 random_state = 42
        )
        model.fit(X_train, y_train)
        print("model.type:", type(model))
        print("model:", model)

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
        mlflow.xgboost.log_model(model, "model", registered_model_name=model_name)


@click.command()
@click.option("--experiment-name",
    help="Experiment name.",
    type=str,
    default=None,
    show_default=True
)
@opt_data_path
@click.option("--model-name",
    help="Registered model name.",
    type=str,
    default=None,
    show_default=True
)
@click.option("--max-depth",
    help="Max depth parameter.",
    type=int,
    default=None,
    show_default=True
)
@click.option("--min-child-weight",
    help="Max leaf nodes parameter.",
    type=float,
    default=1.5,
    show_default=True
)
@click.option("--estimators",
    help="Estimators",
    type=int,
    default=10,
    show_default=True
)
def main(experiment_name, data_path, max_depth, min_child_weight, estimators, model_name):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    model_name = None if not model_name or model_name == "None" else model_name
    train(data_path, max_depth, min_child_weight, estimators, model_name)


if __name__ == "__main__":
    main()
