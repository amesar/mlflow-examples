import pandas as pd
import click
import mlflow
import mlflow.pyfunc
import utils

utils.display_versions()

@click.command()
@click.option("--model-uri", help="Model URI", required=True, type=str)
@click.option("--data-path", help="Data path", default=None, type=str)
def main(model_uri, data_path):
    print("Options:")
    for k,v in locals().items(): print(f"  {k}: {v}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model:", model)

    ndarray = utils.get_prediction_data(data_path)
    data = pd.DataFrame(ndarray)
    print("data.shape:", data.shape)

    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)

if __name__ == "__main__":
    main()
