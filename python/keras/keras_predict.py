from argparse import ArgumentParser
import mlflow
import mlflow.keras
import utils

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

import numpy as np

def load():
    from PIL import Image
    path = "/Users/ander/data/mnist/mnist_png/testing/0/10.png"
    img = Image.open(path).convert("L")
    img = np.resize(img, (28,28,1))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)
    return im2arr 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="model_uri", default="../../data/wine-quality-white.csv")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        
    model = mlflow.keras.load_model(args.model_uri)
    print("model:", type(model))
    
    _,_,data,_  = utils.build_data()
    print(">> data.shape:", data.shape)

    #data2 = load()
    #print("data2.shape:", data2.shape)
    #data = data2

    predictions = model.model.predict_classes(data)
    print("predictions.type:",type(predictions))
    print("predictions.shape:",predictions.shape)
    print("predictions:", predictions)
