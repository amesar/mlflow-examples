from argparse import ArgumentParser
import mlflow
import mlflow.onnx
import utils
import onnx_utils

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="model_uri", default="../../data/train/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    _,_,data,_  = utils.build_data()
    model = mlflow.onnx.load_model(args.model_uri)
    print("model.type:", type(model))

    predictions = onnx_utils.score_model(model, data)
    print("predictions.type:",type(predictions))
    print("predictions.shape:",predictions.shape)
    print("predictions:",predictions)
