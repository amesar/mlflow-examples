import pandas as pd
import mlflow

def build_mnist_data():
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

def build_data(data_path):
    from sklearn.model_selection import train_test_split
    colLabel = "quality"
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.30, random_state=42)

    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train = train.drop([colLabel], axis=1)
    X_test = test.drop([colLabel], axis=1)
    y_train = train[[colLabel]]
    y_test = test[[colLabel]]

    return X_train, X_test, y_train, y_test


def dump(run_id, client = mlflow.tracking.MlflowClient()):
    #toks = model_uri.split("/")
    #run_id = toks[1]
    print("  run_id:",run_id)
    run = client.get_run(run_id)
    exp = client.get_experiment(run.info.experiment_id)
    print("Run:")
    #print("  model_uri:",model_uri)
    print("  run_id:",run_id)
    print("  experiment_id:",exp.experiment_id)
    print("  experiment_name:",exp.name)

def register_model(run, model_name, client = mlflow.tracking.MlflowClient()):
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass
    source = f"{run.info.artifact_uri}/model"
    client.create_model_version(model_name, source, run.info.run_id)
