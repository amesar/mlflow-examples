import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras
from tabulate import tabulate
import utils

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())
client = mlflow.tracking.MlflowClient()
tmp_dir = "out" # TODO 

def predict_keras(model_uri, data):
    print(f"\nmlflow.keras.load_model - {model_uri}")
    model = mlflow.keras.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.predict(data)
    display(predictions)

def predict_pyfunc(model_uri, data):
    print(f"\nmlflow.pyfunc.load_model - {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.predict(data)
    display(predictions)

def predict_onnx(model_uri, data):
    print(f"\nmlflow.onnx.load_model - {model_uri}")
    import mlflow.onnx
    import onnx_utils
    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))
    data = data.to_numpy()
    predictions = onnx_utils.score_model(model, data)
    display(predictions)

def predict_tensorflow_model(run_id, data):
    model_name = "tensorflow-model"
    print(f"\nkeras.models.load_model - {model_name} - {run_id}")
    client.download_artifacts(run_id, model_name, tmp_dir)
    model = keras.models.load_model(os.path.join(tmp_dir, model_name))
    print("model.type:",type(model))
    predictions = model.predict(data)
    display(predictions)

def predict_tensorflow_lite_model(run_id, data):
   # Get model from MLflow
    model_name = "tensorflow-lite-model"
    print(f"\ntf.lite.Interpreter - {model_name} - {run_id}")
    client.download_artifacts(run_id, model_name, tmp_dir)
    path = os.path.join(tmp_dir, model_name, "model.tflite")
    with open(path, "rb") as f:
        model = f.read()
    print("model.type:",type(model))

   # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

   # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("input_details",input_details)
    print("output_details",output_details)

    # Score data - tflite model can only score one data element
    predictions = []
    for x in data.to_numpy():
        input_data = np.array([x])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        p = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(p[0])
    display(np.asarray(predictions))

def display(predictions):
    print("predictions.shape:",predictions.shape)
    df = pd.DataFrame(data=predictions, columns=["prediction"])
    df = df.head(10)
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

def artifact_exists(run_id, path):
    return len(client.list_artifacts(run_id, path)) > 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="run_id", required=True)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default="../../data/train/wine-quality-white.csv")
    parser.add_argument("--score_as_pyfunc", dest="score_as_pyfunc", help="Score as PyFunc", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    run_id = args.run_id

    utils.dump(run_id)
    data,_,_,_  = utils.build_wine_data(args.data_path)

    model_uri = f"runs:/{run_id}/keras-hd5-model"
    predict_keras(model_uri, data)
    if args.score_as_pyfunc:
        predict_pyfunc(model_uri, data)

    model_name = "onnx-model"
    if artifact_exists(run_id, model_name):
        model_uri = f"runs:/{run_id}/{model_name}"
        predict_onnx(model_uri, data)
        predict_pyfunc(model_uri, data)
    else:
        print(f"No model: {model_name}")

    model_name = "tensorflow-model"
    if artifact_exists(run_id, model_name):
        predict_tensorflow_model(run_id, data)
    else:
        print(f"No model: {model_name}")

    model_name = "tensorflow-lite-model"
    if artifact_exists(run_id, model_name):
        predict_tensorflow_lite_model(run_id, data)
    else:
        print(f"No model: {model_name}")
