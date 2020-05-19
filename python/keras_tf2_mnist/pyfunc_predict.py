from argparse import ArgumentParser
import pandas as pd
import click
import mlflow
import mlflow.pyfunc
import utils

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

@click.command()
@click.option("--model_uri", help="Model URI", required=True, type=str)
def main(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    print("model:", model)

    _,_,ndarray,_  = utils.build_data()
    data = pd.DataFrame(ndarray)
    print("data.shape:", data.shape)

    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)

if __name__ == "__main__":
    main()
