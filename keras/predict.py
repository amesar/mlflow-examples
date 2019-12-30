from argparse import ArgumentParser
import mlflow
import mlflow.keras
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
        
    model = mlflow.keras.load_model(args.model_uri)
    print("model:", model)
    
    _,_,data,_  = utils.build_data()
    predictions = model.model.predict_classes(data)
    print("predictions:", predictions)
