# https://catboost.ai/docs/concepts/apply-onnx-ml.html

from argparse import ArgumentParser
import numpy as np
import mlflow
import mlflow.onnx
import onnxruntime as rt
import utils

print("Tracking URI:", mlflow.tracking.get_tracking_uri())
print("MLflow Version:", mlflow.__version__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="model_uri", required=True)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../../data/train/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    X,y  = utils.build_data(args.data_path)
    model = mlflow.onnx.load_model(args.model_uri)
    print("model:",type(model))
    session = rt.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: X.to_numpy().astype(np.float32)})[0]
    print("predictions:",predictions)
