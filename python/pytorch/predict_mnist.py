from argparse import ArgumentParser
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import utils

print("Torch Version:", torch.__version__)
client = mlflow.tracking.MlflowClient()
print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="Run ID", required=True)
    parser.add_argument("--score_as_pyfunc", dest="score_as_pyfunc", help="Score as Pyfunc", default=False, action='store_true')
    parser.add_argument("--score_as_onnx", dest="score_as_onnx", help="Score as ONNX", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    print("\n**** Data")
    loader = utils.get_data(False,10000)
    print("loader.type:", type(loader))
    data = utils.prep_data(loader)
    print("data.type:", type(data))
    print("data.shape:", data.shape)

    print("\n**** pytorch.load_model")

    model_uri = f"runs:/{args.run_id}/pytorch-model"
    model = mlflow.pytorch.load_model(model_uri)
    print("model.type:", type(model))

    outputs = model(data)
    print("outputs.type:", type(outputs))
    outputs = outputs.detach().numpy()
    print("outputs.shape:",outputs.shape)
    utils.display_predictions(outputs)

    # TODO: convert tensor to Pyfunc scoring format
    if args.score_as_pyfunc:
        print("\n**** pyfunc.load_model")
        model = mlflow.pyfunc.load_model(model_uri)
        print("model.type:", type(model))
        data_pd = pd.DataFrame(data.numpy()) #  TODO: ValueError: Must pass 2-d input
        outputs = model.predict(data_pd)
        print("outputs.type:", type(outputs))

    if args.score_as_onnx:
        print("\n**** onnx.load_model - onnx\n")
        import mlflow.onnx
        import onnx
        import onnx_utils
        print("ONNX Version:", onnx.__version__)

        model_uri = f"runs:/{args.run_id}/onnx-model"
        model = mlflow.onnx.load_model(model_uri)
        print("model.type:", type(model))

        # TODO: convert tensor to ONNX scoring format
        # INVALID_ARGUMENT : Got invalid dimensions for input: input.1 for the following indices
        # index: 0 Got: 10000 Expected: 64
        #data = data.numpy() 
        print(">> data.type:",type(data))
        data = to_numpy(data)
        print(">> data.type:",type(data))

        outputs = onnx_utils.score_model(model, data)
        print("outputs.type:", type(outputs))
        print("outputs:\n",  pd.DataFrame(outputs))

