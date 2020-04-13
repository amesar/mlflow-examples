from argparse import ArgumentParser
import pandas as pd
import mlflow
import mlflow.pyfunc

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="Model URI", required=True)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default="../../data/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    model = mlflow.pyfunc.load_model(args.model_uri)
    print("model:", model)

    data = pd.read_csv(args.data_path)
    print("data.shape:",data.shape)

    predictions = model.predict(data)
    print("predictions:", predictions[:5])
    print("predictions.len:", len(predictions))

