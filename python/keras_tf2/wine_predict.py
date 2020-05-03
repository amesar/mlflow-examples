from argparse import ArgumentParser
import pandas as pd
import mlflow
import mlflow.keras
import utils
from tabulate import tabulate

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

def predict_keras(model_uri, data):
    print(f"\nmlflow.keras.load_model - {model_uri}")
    model = mlflow.keras.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.predict(data)
    display(predictions)

def predict_pyfunc(model_uri, data):
    print(f"\nmlflow.pyfunc.load_model - {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.predict(data)
    display(predictions)

def predict_onnx(model_uri, data):
    print(f"\nmlflow.onnx.load_model - {model_uri}")
    import mlflow.onnx
    import onnx_utils
    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))
    data = data.to_numpy()
    predictions = onnx_utils.score_model(model, data)
    display(predictions)

def display(predictions):
    print("predictions.shape:",predictions.shape)
    df = pd.DataFrame(data=predictions, columns=["prediction"])
    df = df.head(10)
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="run_id", required=True)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default="../../data/wine-quality-white.csv")
    parser.add_argument("--score_as_pyfunc", dest="score_as_pyfunc", help="Score as PyFunc", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    utils.dump(args.run_id)
    data,_,_,_  = utils.build_wine_data(args.data_path)

    model_uri = f"runs:/{args.run_id}/keras-model"
    predict_keras(model_uri, data)
    if args.score_as_pyfunc:
        predict_pyfunc(model_uri, data)

    client = mlflow.tracking.MlflowClient()
    onnx_model = "onnx-model"
    try:
        client.download_artifacts(args.run_id, onnx_model, "tmp")
        model_uri = f"runs:/{args.run_id}/{onnx_model}"
        predict_onnx(model_uri, data)
        predict_pyfunc(model_uri, data)
    except Exception as e:
        print(f"NO ONNX model: {onnx_model}")

