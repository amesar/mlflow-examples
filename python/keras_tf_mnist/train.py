import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import mlflow
import mlflow.keras
import mlflow.tensorflow
import click
import utils

utils.display_versions()

np.random.seed(42)
tf.random.set_seed(42)

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model

def train(run, model_name, data_path, epochs, batch_size, mlflow_custom_log, log_as_onnx):
    x_train, y_train, x_test, y_test = utils.get_train_data(data_path)
    model = build_model()

    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    print("model.type:",type(model))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("test_acc:", test_acc)
    print("test_loss:", test_loss)

    if mlflow_custom_log:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_loss", test_loss)

        # Save as TensorFlow SavedModel flavor
        mlflow.keras.log_model(model, "keras-model-tf", registered_model_name=f"{model_name}", save_format="tf")

        # Save as H5 format (MLflow Keras default)
        mlflow.keras.log_model(model, "keras-model-h5")

        # write model summary
        summary = []
        model.summary(print_fn=summary.append)
        summary = "\n".join(summary)
        with open("model_summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("model_summary.txt")

    else:
        utils.register_model(run, model_name)

    # write model as yaml file
    with open("model.yaml", "w") as f:
        f.write(model.to_yaml())
    mlflow.log_artifact("model.yaml")

    # MLflow - log onnx model
    if log_as_onnx:
        import onnx_utils
        mname = f"{model_name}_onnx" if model_name else None
        onnx_utils.log_model(model, "onnx-model", mname)

    predictions = model.predict_classes(x_test)
    print("predictions:", predictions)

@click.command()
@click.option("--experiment_name", help="Experiment name", default=None, type=str)
@click.option("--model_name", help="Registered model name", default=None, type=str)
@click.option("--data_path", help="Data path", default=None, type=str)
@click.option("--epochs", help="Epochs", default=5, type=int)
@click.option("--batch_size", help="Batch size", default=128, type=int)
@click.option("--repeats", help="Repeats", default=1, type=int)
@click.option("--mlflow_custom_log", help="Log params/metrics with mlflow.log", default=True, type=bool)
@click.option("--keras_autolog", help="Automatically log params/ metrics with mlflow.keras.autolog", default=False, type=bool)
@click.option("--tensorflow_autolog", help="Automatically log params/ metrics with mlflow.tensorflow.autolog", default=False, type=bool)
@click.option("--log_as_onnx", help="log_as_onnx", default=False, type=bool)

def main(experiment_name, model_name, data_path, epochs, batch_size, repeats, keras_autolog, tensorflow_autolog, mlflow_custom_log, log_as_onnx):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    model_name = None if not model_name or model_name == "None" else model_name

    if keras_autolog:
        mlflow.keras.autolog()
    if tensorflow_autolog:
        mlflow.tensorflow.autolog()

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    for i in range(0,repeats):
        with mlflow.start_run() as run:
            print(f"******** {i}/{repeats}")
            print("MLflow:")
            print("  run_id:",run.info.run_id)
            print("  experiment_id:",run.info.experiment_id)
            mlflow.set_tag("version.mlflow", mlflow.__version__)
            mlflow.set_tag("version.keras", keras.__version__)
            mlflow.set_tag("version.tensorflow", tf.__version__)
            mlflow.set_tag("keras_autolog", keras_autolog)
            mlflow.set_tag("tensorflow_autolog", tensorflow_autolog)
            mlflow.set_tag("mlflow_custom_log", mlflow_custom_log)
            train(run, model_name, data_path, epochs, batch_size, mlflow_custom_log, log_as_onnx)

if __name__ == "__main__":
    main()
