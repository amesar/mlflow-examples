import click
import pandas as pd
import mlflow
import mlflow.pyfunc
import common

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

@click.command()
@click.option("--model-uri", help="Model URI", default=None, type=str)
@click.option("--data-path", help="Data path", default=common.default_data_path, type=str)

def main(model_uri, data_path):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")

    data = pd.read_csv(data_path)
    print("data.shape:",data.shape)

    model = mlflow.pyfunc.load_model(model_uri)
    print("model:", model)
    print("model.type:", type(model))

    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.len:", len(predictions))
    print("predictions:", predictions[:5])

if __name__ == "__main__":
    main()
