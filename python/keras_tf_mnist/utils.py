import pandas as pd
import numpy as np
import mlflow

def reshape(x, n):
    x = x.reshape((n, 28 * 28))
    return x.astype('float32') / 255

def get_train_data(data_path=None):
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    if data_path:
        raise NotImplementedError(f"Custom data file yet supported: {data_path}")
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Data:")
    print("  x_train.shape:", x_train.shape)
    print("  y_train.shape:", y_train.shape)
    print("  x_test.shape:", x_test.shape)
    print("  y_test.shape:", y_test.shape)

    x_train = reshape(x_train, x_train.shape[0])
    x_test = reshape(x_test, x_test.shape[0])
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print("Data after reshape:")
    print("  x_train.shape:", x_train.shape)
    print("  y_train.shape:", y_train.shape)
    print("  x_test.shape:", x_test.shape)
    print("  y_test.shape:", y_test.shape)

    return x_train, y_train, x_test, y_test

def get_prediction_data(data_path=None):
    if not data_path:
        _,_,x_test,_  = get_train_data()
        data = x_test
    elif data_path.endswith(".json"):
        data = pd.read_json(data_path, orient="split")
    elif data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    elif data_path.endswith(".npz"):
        with np.load(data_path) as data:
            data = data["x_test"]
        data = reshape(data, 10000)
    elif data_path.endswith(".png"):
        from PIL import Image
        nparray = np.asarray(Image.open(data_path))
        data = nparray.reshape((1, 28 * 28))
    else:
        raise Exception(f"Unknown file extension '{data_path}'")
    return data

def register_model(run, model_name, client = mlflow.tracking.MlflowClient()):
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException: # RESOURCE_ALREADY_EXISTS
        pass
    source = f"{run.info.artifact_uri}/model"
    client.create_model_version(model_name, source, run.info.run_id)

def predict_pyfunc(model_uri, data):
    print("\n**** mlflow.pyfunc.load_model\n")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    data = pd.DataFrame(data)
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    #print("predictions:", predictions)
    display_predictions(predictions)

def display_predictions(data):
    from tabulate import tabulate
    df = pd.DataFrame(data).head(10)
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

def display_versions():
    import tensorflow as tf
    import tensorflow.keras as keras
    import platform
    print("Versions:")
    print("  MLflow Version:", mlflow.__version__)
    print("  TensorFlow version:", tf.__version__)
    print("  Keras version:", keras.__version__)
    print("  Python Version:", platform.python_version())
    print("  Operating System:", platform.system()+" - "+platform.release())
    print("  Tracking URI:", mlflow.tracking.get_tracking_uri())

