from argparse import ArgumentParser
import torch
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import utils

print("Torch Version:", torch.__version__)
client = mlflow.tracking.MlflowClient()
print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="Model URI", required=True)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    loader = utils.get_data(False,10000)
    print("loader.type:", type(loader))
    data = utils.prep_data(loader)
    print("data.type:", type(data))
    print("data.shape:", data.shape)

    print("==== pytorch.load_model")

    model = mlflow.pytorch.load_model(args.model_uri)
    print("model.type:", type(model))

    outputs = model(data)
    print("outputs.type:", type(outputs))
    outputs = outputs.detach().numpy()
    print("outputs.shape:",outputs.shape)
    utils.display_predictions(outputs)
