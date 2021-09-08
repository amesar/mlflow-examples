import click
import pandas as pd
import mlflow
import mlflow.onnx
import onnx_utils

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

@click.command()
@click.option("--model-uri", help="Model URI", default=None, type=str)
@click.option("--data-path", help="Data path", default="../../data/train/wine-quality-white.csv", type=str)

def main(model_uri, data_path):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")

    data = pd.read_csv(data_path).to_numpy()

    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))

    predictions = onnx_utils.score_model(model, data)
    print("predictions.type:",type(predictions))
    print("predictions.shape:",predictions.shape)
    print("predictions:",predictions)

if __name__ == "__main__":
    main()
