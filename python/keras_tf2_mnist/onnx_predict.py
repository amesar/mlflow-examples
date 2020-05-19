import pandas as pd
import click
import mlflow
import mlflow.pyfunc
import mlflow.onnx
import utils
import onnx_utils

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

@click.command()
@click.option("--model_uri", help="Model URI", required=True, type=str)
def main(model_uri):
    print("model_uri:", model_uri)
    _,_,data,_  = utils.build_data()

    print("\n**** mlflow.onnx.load_model\n")
    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))
    predictions = onnx_utils.score_model(model, data)
    print("predictions.type:",type(predictions))
    print("predictions.shape:",predictions.shape)
    print("predictions:",predictions)

    utils.predict_pyfunc(model_uri, data)

def foo():
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
