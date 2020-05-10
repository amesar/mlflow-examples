"""
Train Wine Quality ataset with KerasRegressor
"""

import platform
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import mlflow
import mlflow.keras
import utils

print("Versions:")
print("  Tracking URI:", mlflow.tracking.get_tracking_uri())
print("  MLflow Version:", mlflow.__version__)
print("  Keras version:", keras.__version__)
print("  TensorFlow version:", tf.__version__)
print("  Operating System:",platform.system()+" - "+platform.release())

np.random.seed(42)
tf.random.set_seed(42)

tmp_dir = "out"

def train(data_path, epochs, batch_size, mlflow_log, log_as_onnx):
    print("mlflow_log:", mlflow_log)
    x_train, _, y_train, _ = utils.build_wine_data(data_path)

    ncols = x_train.shape[1]
    def baseline_model():
        model = Sequential()
        model.add(Dense(ncols, input_dim=ncols, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    model = baseline_model()

    if mlflow_log:
        print("Logging with mlflow.log")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

    # MLflow - log Keras HD5 model
    mlflow.keras.log_model(model, "keras-hd5-model", registered_model_name=args.model_name)

    # MLflow - log onnx model
    if log_as_onnx:
        import onnx_utils
        onnx_utils.log_model(model, "onnx-model")

    # Save as TensorFlow SavedModel format
    path = "tensorflow-model"
    tf.keras.models.save_model(model, path, overwrite=True, include_optimizer=True)
    mlflow.log_artifact(path)

    # Save as TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path = "model.tflite"
    with open(path, "wb") as f:
        f.write(tflite_model)
    mlflow.log_artifact(path, "tensorflow-lite-model")

    # Save as TensorFlow.js format
    try:
        import tensorflowjs as tfjs
        path = "model.js"
        tfjs.converters.save_keras_model(model, path)
        mlflow.log_artifact(path, "tensorflow-js")
    except ModuleNotFoundError:
        print("WARNING: tensorflowjs is not installed")

    # Evaluate model
    estimator = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=0)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, x_train, y_train, cv=kfold)
    print(f"Baseline MSE: mean: {round(results.mean(),2)}  std: {round(results.std(),2)}")
    if mlflow_log:
        mlflow.log_metric("mse_mean", results.mean())
        mlflow.log_metric("mse_std", results.std())

    # Score
    data = x_train
    predictions = model.predict(data)
    predictions = pd.DataFrame(data=predictions, columns=["prediction"])
    print("predictions.shape:",predictions.shape)
    print("predictions:",predictions)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment name", required=False, type=str)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default="../../data/train/wine-quality-white.csv")
    parser.add_argument("--model_name", dest="model_name", help="Registered model name", default=None)
    parser.add_argument("--epochs", dest="epochs", help="Epochs", default=5, type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="Batch size", default=128, type=int)
    parser.add_argument("--mlflow_log", dest="mlflow_log", help="Log params/metrics with mlflow.log", default=False, type=bool)
    parser.add_argument("--keras_autolog", dest="keras_autolog", help="Automatically log params/ metrics with mlflow.keras.autolog", default=False, type=bool)
    parser.add_argument("--tensorflow_autolog", dest="tensorflow_autolog", help="Automatically log params/ metrics with mlflow.tensorflow.autolog", default=False, type=bool)
    parser.add_argument("--log_as_onnx", dest="log_as_onnx", help="Log model as ONNX flavor", default=False, type=bool)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    if args.keras_autolog:
        print("Logging with mlflow.keras.autolog")
        mlflow.keras.autolog()
    if args.tensorflow_autolog:
        print("Logging with mlflow.tensorflow.autolog")
        import mlflow.tensorflow
        mlflow.tensorflow.autolog()

    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run() as run:
        print("MLflow:")
        print("  run_id:",run.info.run_id)
        print("  experiment_id:",run.info.experiment_id)

        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.keras", keras.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)
        mlflow.set_tag("version.sklearn", sklearn.__version__)
        mlflow.set_tag("version.os", platform.system()+" - "+platform.release())
        mlflow.set_tag("keras_autolog", args.keras_autolog)
        mlflow.set_tag("tensorflow_autolog", args.tensorflow_autolog)

        train(args.data_path, args.epochs, args.batch_size, args.mlflow_log, args.log_as_onnx)
