
from argparse import ArgumentParser
import pandas as pd
import torch
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.onnx
import onnx_utils

print("Torch Version:", torch.__version__)
client = mlflow.tracking.MlflowClient()
print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="run_id", required=True)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    data = torch.Tensor([[1.0], [2.0], [3.0]])
    data_pd = pd.DataFrame(data.numpy())
    print("data.type:", type(data))
    print("data.shape:", data.shape)

    print("==== pytorch.load_model\n")
    model_uri = f"runs:/{args.run_id}/pytorch-model"
    print("model_uri:",model_uri)

    model = mlflow.pytorch.load_model(model_uri)
    #print("model:", model)
    print("model.type:", type(model))

    outputs = model(data)
    print("outputs.type:", type(outputs))
    outputs = outputs.detach().numpy()
    outputs = pd.DataFrame(outputs)
    print("outputs:\n", outputs)

    print("\n==== pyfunc.load_model - pytorch\n")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    outputs = model.predict(data_pd)
    print("outputs.type:", type(outputs))
    print("outputs:\n", outputs)

    artifacts = client.list_artifacts(args.run_id, "onnx-model")
    if len(artifacts) > 0:
        model_uri = f"runs:/{args.run_id}/onnx-model"
        print("\n==== onnx.load_model - onnx\n")
        model = mlflow.onnx.load_model(model_uri)
        print("model.type:", type(model))
        outputs = onnx_utils.score_model(model, data_pd.to_numpy())
        print("outputs.type:", type(outputs))
        print("outputs:\n",  pd.DataFrame(outputs))

        print("\n==== pyfunc.load_model - onnx\n")
        print("model_uri:",model_uri)
        model = mlflow.pyfunc.load_model(model_uri)
        print("model.type:", type(model))
        outputs = model.predict(data_pd)
        print("outputs.type:", type(outputs))
        print("outputs:\n", outputs)

