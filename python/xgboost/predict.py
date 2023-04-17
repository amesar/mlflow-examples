import click
import pandas as pd
import xgboost as xgb
import mlflow
from common import opt_data_path

print("Tracking URI:", mlflow.tracking.get_tracking_uri())
print("MLflow Version:", mlflow.__version__)
print("XGBoost version:", xgb.__version__)


def build_data(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(["quality"], axis=1)
    return X


def predict_xgboost(model_uri, X):
    print("\n=== mlflow.xgboost.load_model\n")
    model = mlflow.xgboost.load_model(model_uri)
    print("model.type:", type(model))
    print("model:", model)
    predictions = model.predict(X)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)


def predict_pyfunc(model_uri, X):
    print("\n=== mlflow.pyfunc.load_model\n")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    print("model:", model)
    predictions = model.predict(X)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)


@click.command
@click.option("--model-uri",
    help="Model URI.",
    type=str,
    required=True
)
@opt_data_path
def main(model_uri, data_path):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    X = build_data(data_path)
    predict_pyfunc(model_uri, X)
    predict_xgboost(model_uri, X)


if __name__ == "__main__":
    main()
