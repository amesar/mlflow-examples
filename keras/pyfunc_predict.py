from argparse import ArgumentParser
import pandas as pd
import mlflow
import mlflow.pyfunc
import utils

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="model_uri", default="../data/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    model = mlflow.pyfunc.load_pyfunc(args.model_uri)
    print("model:", model)

    _,_,ndarray,_  = utils.build_data()
    data = pd.DataFrame(ndarray)

    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)
