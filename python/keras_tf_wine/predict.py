import os
import click
import platform
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
print("Operating System:",platform.system()+" - "+platform.release())
tmp_dir = "out" # TODO 

#def predict_keras(model_uri, data):
def predict_tensorflow_model(model_uri, data):
    print(f"\nmlflow.keras.load_model\nModel URI: {model_uri}")
    model = mlflow.keras.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.predict(data)
    display(predictions)

def _predict_tensorflow_model(run_id, data):
    model_name = "tensorflow-model"
    print(f"\nkeras.models.load_model\nModel name:{model_name}\nRun ID:{run_id}")
    client.download_artifacts(run_id, model_name, tmp_dir)
    model = keras.models.load_model(os.path.join(tmp_dir, model_name))
    print("model.type:",type(model))
    predictions = model.predict(data)
    display(predictions)

def predict_pyfunc(model_uri, data, msg):
    print(f"\nmlflow.pyfunc.load_model - {msg}\nModel URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.predict(data)
    display(predictions)

def predict_onnx(model_uri, data):
    print(f"\nmlflow.onnx.load_model\nModel URI: {model_uri}")
    import mlflow.onnx
    import onnx_utils
    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))
    data = data.to_numpy()
    predictions = onnx_utils.score_model(model, data)
    display(predictions)

def predict_tensorflow_lite_model(run_id, data):
   # Get model from MLflow
    model_name = "tensorflow-lite-model"
    print(f"\ntf.lite.Interpreter\nModel name:{model_name}\nRun ID:{run_id}")
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
    df = df.head(2)
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

def artifact_exists(run_id, path):
    return len(client.list_artifacts(run_id, path)) > 0

@click.command()
@click.option("--run_id", help="RunID", default=None, type=str)
@click.option("--data_path", help="Data path", default="../../data/train/wine-quality-white.csv", type=str)
@click.option("--score_as_pyfunc", help="Score as PyFunc", default=True, type=bool)
@click.option("--score_as_tensorflow_lite", help="Score as TensorFlow Lite", default=False, type=bool)

def main(run_id, data_path, score_as_pyfunc, score_as_tensorflow_lite):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")

    utils.dump(run_id)
    data,_,_,_  = utils.build_data(data_path)

    model_uri = f"runs:/{run_id}/tensorflow-model"
    predict_tensorflow_model(model_uri, data)
    if score_as_pyfunc:
        predict_pyfunc(model_uri, data, "tensorflow-model")

    if score_as_tensorflow_lite:
        model_name = "tensorflow-lite-model"
        if artifact_exists(run_id, model_name):
            predict_tensorflow_lite_model(run_id, data)
        else:
            print(f"WARNING: no model '{model_name}'")

    model_name = "onnx-model"
    if artifact_exists(run_id, model_name):
        model_uri = f"runs:/{run_id}/{model_name}"
        predict_onnx(model_uri, data)
        predict_pyfunc(model_uri, data, "onnx-model")
    else:
        print(f"WARNING: no model '{model_name}'")


if __name__ == "__main__":
    main()
