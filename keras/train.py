"""
Adapted from: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/2.1-a-first-look-at-a-neural-network.ipynb
"""

import numpy as np
import tensorflow as tf
import keras
import mlflow
import mlflow.keras
import utils

print("Tracking URI:", mlflow.tracking.get_tracking_uri())
print("MLflow Version:", mlflow.version.VERSION)
print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)

np.random.seed(42)
tf.set_random_seed(42)

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def train(epochs, batch_size, autolog):

    x_train, y_train, x_test, y_test = utils.build_data()
    model = build_model()

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("test_acc:", test_acc)
    print("test_loss:", test_loss)

    if not autolog:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.keras.log_model(model, "keras-model")

        # write model as yaml file
        with open("model.yaml", "w") as f:
            f.write(model.to_yaml())
        mlflow.log_artifact("model.yaml")

        # write model summary
        summary = []
        model.summary(print_fn=summary.append)
        summary = '\n'.join(summary)
        with open("model_summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("model_summary.txt")

    predictions = model.predict_classes(x_test)
    print("predictions:", predictions)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default="mnist_keras")
    parser.add_argument("--epochs", dest="epochs", help="epochs", default=5, type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="batch_size", default=128, type=int)
    parser.add_argument("--repeats", dest="repeats", help="repeats", default=1, type=int)

    parser.add_argument("--autolog", dest="autolog", help="Automatically log params and metrics", default=False, action='store_true')

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    if args.autolog:
        mlflow.keras.autolog()

    mlflow.set_experiment(args.experiment_name)
    for i in range(0,args.repeats):
        with mlflow.start_run() as run:
            print(f"******** {i}/{args.repeats}")
            print("MLflow:")
            print("  run_id:",run.info.run_id)
            print("  experiment_id:",run.info.experiment_id)
            mlflow.set_tag("mlflow_version", mlflow.version.VERSION)
            mlflow.set_tag("keras_version", keras.__version__)
            mlflow.set_tag("tensorflow_version", tf.__version__)
            mlflow.set_tag("autolog", args.autolog)
            train(args.epochs, args.batch_size, args.autolog)
