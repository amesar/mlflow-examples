import click
import pandas as pd
import mlflow
import mlflow.pyfunc

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

@click.command()
@click.option("--model_uri", help="Model URI", default=None, type=str)
@click.option("--data_path", help="Data path", default="../../data/train/wine-quality-white.csv", type=str)

def main(model_uri, data_path):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")

    model = mlflow.pyfunc.load_model(model_uri)
    print("model:", model)

    data = pd.read_csv(data_path)
    print("data.shape:",data.shape)

    predictions = model.predict(data)
    print("predictions:", predictions[:5])
    print("predictions.len:", len(predictions))

if __name__ == "__main__":
    main()
