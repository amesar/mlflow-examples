from argparse import ArgumentParser
import numpy as np
import mlflow
import mlflow.keras
import utils

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="model_uri", default="../../data/train/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        
    model = mlflow.keras.load_model(args.model_uri)
    print("model:", type(model))
    
    _,_,data,_  = utils.build_data()
    print("data.type:", type(data))
    print("data.shape:", data.shape)

    print("== model.predict")
    predictions = model.predict(data)
    print("predictions.type:",type(predictions))
    print("predictions.shape:",predictions.shape)
    print("predictions:", predictions)

    print("== model.predict_classes")
    predictions = model.predict_classes(data)
    print("predictions:", predictions)
