from argparse import ArgumentParser
import pandas as pd
import mlflow
import mlflow.onnx
import onnx_utils

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="Model URI", required=True)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default="../data/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    data = pd.read_csv(args.data_path).to_numpy()

    model = mlflow.onnx.load_model(args.model_uri)
    print("model.type:", type(model))

    predictions = onnx_utils.score_model(model, data)
    print("predictions.type:",type(predictions))
    print("predictions.shape:",predictions.shape)
    print("predictions:",predictions)
