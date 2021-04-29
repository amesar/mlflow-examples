"""
Train Wine Quality dataset with KerasRegressor
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
import click
import utils

print("Versions:")
print("  Tracking URI:", mlflow.tracking.get_tracking_uri())
print("  MLflow Version:", mlflow.__version__)
print("  Keras version:", keras.__version__)
print("  TensorFlow version:", tf.__version__)
print("  Operating System:",platform.system()+" - "+platform.release())

np.random.seed(42)
tf.random.set_seed(42)

def train(run, model_name, data_path, epochs, batch_size, mlflow_custom_log, log_as_onnx, log_as_tensorflow_lite, log_as_tensorflow_js):
    print("mlflow_custom_log:", mlflow_custom_log)
    x_train, _, y_train, _ = utils.build_data(data_path)

    ncols = x_train.shape[1]
    def baseline_model():
        model = Sequential()
        model.add(Dense(ncols, input_dim=ncols, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    model = baseline_model()

    if mlflow_custom_log:
        print("Logging with mlflow.log")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.keras.log_model(model, "tensorflow-model", registered_model_name=model_name)
    else:
        utils.register_model(run, model_name)

    # MLflow - log as ONNX model
    if log_as_onnx:
        import onnx_utils
        mname = f"{model_name}_onnx" if model_name else None
        onnx_utils.log_model(model, "onnx-model", model_name=mname)

    # Save as TensorFlow Lite format
    if log_as_tensorflow_lite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        path = "model.tflite"
        with open(path, "wb") as f:
            f.write(tflite_model)
        mlflow.log_artifact(path, "tensorflow-lite-model")

    # Save as TensorFlow.js format
    if log_as_tensorflow_js:
        import tensorflowjs as tfjs
        path = "model.tfjs"
        tfjs.converters.save_keras_model(model, path)
        mlflow.log_artifact(path, "tensorflow-js-model")

    # Evaluate model
    estimator = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=0)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, x_train, y_train, cv=kfold)
    print(f"Baseline MSE: mean: {round(results.mean(),2)}  std: {round(results.std(),2)}")
    if mlflow_custom_log:
        mlflow.log_metric("mse_mean", results.mean())
        mlflow.log_metric("mse_std", results.std())

    # Score
    data = x_train
    predictions = model.predict(data)
    predictions = pd.DataFrame(data=predictions, columns=["prediction"])
    print("predictions.shape:",predictions.shape)
    print("predictions:",predictions)

@click.command()
@click.option("--experiment_name", help="Experiment name", default=None, type=str)
@click.option("--data_path", help="Data path", default="../../data/train/wine-quality-white.csv", type=str)
@click.option("--model_name", help="Registered model name", default=None, type=str)
@click.option("--epochs", help="Epochs", default=5, type=int)
@click.option("--batch_size", help="Batch size", default=128, type=int)
@click.option("--mlflow_custom_log", help="Log params/metrics with mlflow.log", default=True, type=bool)
@click.option("--keras_autolog", help="Automatically log params/ metrics with mlflow.keras.autolog", default=False, type=bool)
@click.option("--tensorflow_autolog", help="Automatically log params/ metrics with mlflow.tensorflow.autolog", default=False, type=bool)
@click.option("--log_as_onnx", help="log_as_onnx", default=False, type=bool)
@click.option("--log_as_tensorflow_lite", help="log_as_tensorflow_lite", default=False, type=bool)
@click.option("--log_as_tensorflow_js", help="log_as_tensorflow_js", default=False, type=bool)

def main(experiment_name, data_path, model_name, epochs, batch_size, keras_autolog, tensorflow_autolog, mlflow_custom_log, log_as_onnx, log_as_tensorflow_lite, log_as_tensorflow_js):
    import mlflow
    print("Options:")
    for k,v in locals().items(): 
        print(f"  {k}: {v}")
    model_name = None if not model_name or model_name == "None" else model_name

    if keras_autolog:
        print("Logging with mlflow.keras.autolog")
        mlflow.keras.autolog()
    if tensorflow_autolog:
        print("Logging with mlflow.tensorflow.autolog")
        import mlflow.tensorflow
        mlflow.tensorflow.autolog()

    if experiment_name:
        mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        print("MLflow:")
        print("  run_id:",run.info.run_id)
        print("  experiment_id:",run.info.experiment_id)

        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.keras", keras.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)
        mlflow.set_tag("version.sklearn", sklearn.__version__)
        mlflow.set_tag("version.os", platform.system()+" - "+platform.release())
        mlflow.set_tag("mlflow_custom_log", mlflow_custom_log)
        mlflow.set_tag("mlflow_keras.autolog", keras_autolog)
        mlflow.set_tag("mlflow_tensorflow.autolog", tensorflow_autolog)

        train(run, model_name, data_path, epochs, batch_size, mlflow_custom_log, log_as_onnx, log_as_tensorflow_lite, log_as_tensorflow_js)

if __name__ == "__main__":
    main()
