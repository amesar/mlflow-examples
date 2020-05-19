import pandas as pd
import mlflow

def reshape(x, n):
    x = x.reshape((n, 28 * 28))
    x = x.astype('float32') / 255
    return x

def build_data():
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Data:")
    print("  x_train.shape:", x_train.shape)
    print("  y_train.shape:", y_train.shape)
    print("  x_test.shape:", x_test.shape)
    print("  y_test.shape:", y_test.shape)

    x_train = reshape(x_train, 60000)
    x_test = reshape(x_test, 10000)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print("Data after reshape:")
    print("  x_test.shape:", x_test.shape)
    print("  y_test.shape:", y_test.shape)

    return x_train, y_train, x_test, y_test

def register_model(run, model_name, client = mlflow.tracking.MlflowClient()):
    try:
        client.create_registered_model(model_name)
    except Exception:
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
    print("predictions:", predictions)
