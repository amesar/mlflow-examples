from argparse import ArgumentParser
import catboost
import mlflow
import mlflow.sklearn
import utils

print("Tracking URI:", mlflow.tracking.get_tracking_uri())
print("MLflow Version:", mlflow.__version__)
print("Catboost version:",catboost.__version__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-uri", dest="model_uri", help="model_uri", required=True)
    parser.add_argument("--data-path", dest="data_path", help="data_path", default="../../data/train/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    X,y  = utils.build_data(args.data_path)
        
    print("\n=== mlflow.sklearn.load_model")
    model = mlflow.sklearn.load_model(args.model_uri)
    print("model:", type(model))
    predictions = model.predict(X)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)

    print("\n=== mlflow.pyfunc.load_model")
    model = mlflow.pyfunc.load_model(args.model_uri)
    print("model:", type(model))
    predictions = model.predict(X)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)
