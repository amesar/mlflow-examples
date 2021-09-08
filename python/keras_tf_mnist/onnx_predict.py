import pandas as pd
import click
import mlflow
import mlflow.pyfunc
import mlflow.onnx
import utils
import onnx_utils

utils.display_versions()

@click.command()
@click.option("--model-uri", help="Model URI", required=True, type=str)
def main(model_uri):
    print("model_uri:", model_uri)
    data = utils.get_prediction_data()

    print("\n**** mlflow.onnx.load_model\n")
    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))
    predictions = onnx_utils.score_model(model, data)
    print("predictions.type:",type(predictions))
    print("predictions.shape:",predictions.shape)
    print("predictions:",predictions)

    utils.predict_pyfunc(model_uri, data)

    print("\n**** mlflow.pyfunc.load_model\n")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    data = pd.DataFrame(data)
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)

if __name__ == "__main__":
    main()
