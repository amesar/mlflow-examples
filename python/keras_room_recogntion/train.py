import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import mlflow
import mlflow.keras
import mlflow.tensorflow
import click
import utils
from utils import *



np.random.seed(42)
tf.random.set_seed(42)

datadir = "/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/early_exit_model/watch_data/labelled/"
categories = ['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']

img_size = 299
input_shape = (299, 299, 3)

def build_model():
    #model = keras.models.Sequential()
    #model.add(keras.layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    #model.add(keras.layers.Dense(10, activation="softmax"))
    #build model
    dense_layer=4
    layer_size=128
    conv_layer=2

    model = keras.models.Sequential()
    model.add(Conv2D(layer_size,(3,3), input_shape = input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    for l in range(conv_layer):
        model.add(Conv2D(layer_size,(3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    for l in range(dense_layer):
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(categories), activation='softmax'))
  
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def train(run, model_name, data_path, epochs, batch_size, mlflow_custom_log, log_as_onnx):

    channel = 3
    training_data=[]
    create_training_data(categories,datadir,img_size,training_data)
    random.shuffle(training_data)
    X = []
    y = []
    for features,label in training_data:
        X.append(features)
        y.append(label)
    
    print(len(X))
    
    X=np.array(X).reshape(-1,img_size,img_size,channel)  #(cannot pass list directly, -1=(calculates the array size), size,1=gray scale)
    #class_num=keras.utils.np_utils.to_categorical(y,num_classes=len(categories))   #one-hot encoder for cateorical values
    y_class_num=keras.utils.to_categorical(y,num_classes=len(categories))   #one-hot encoder for cateorical values
    
    print('reshape:')
    print(len(X))
    print(X.ndim)

    model = build_model()

    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    model.summary()
    model.fit(X, y_class_num, epochs=epochs, batch_size=32,validation_split=0.2)
    print("model.type:",type(model))

 #   test_loss, test_acc = model.evaluate(x_test, y_test)
 #   print("test_acc:", test_acc)
 #   print("test_loss:", test_loss)

    if mlflow_custom_log:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

#        mlflow.log_metric("test_acc", test_acc)
#        mlflow.log_metric("test_loss", test_loss)

        # Save as TensorFlow SavedModel format (MLflow Keras default)
        mlflow.keras.log_model(model, "keras-model", registered_model_name=model_name)
        #mlflow.keras.log_model(model, "keras-model")

        # write model summary
        summary = []
        model.summary(print_fn=summary.append)
        summary = "\n".join(summary)
        with open("model_summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("model_summary.txt")

    elif model_name:
        utils.register_model(run, model_name)

    # write model as json file
    with open("model.json", "w") as f:
        f.write(model.to_json())
    mlflow.log_artifact("model.json")

    # MLflow - log onnx model
    if log_as_onnx:
        import onnx_utils
        mname = f"{model_name}_onnx" if model_name else None
        onnx_utils.log_model(model, "onnx-model", mname)


@click.command()
@click.option("--experiment-name", help="Experiment name", default=None, type=str)
@click.option("--model-name", help="Registered model name", default=None, type=str)
@click.option("--data-path", help="Data path", default=None, type=str)
@click.option("--epochs", help="Epochs", default=5, type=int)
@click.option("--batch-size", help="Batch size", default=128, type=int)
@click.option("--mlflow-custom-log", help="Explicitly log params, metrics and model with mlflow.log_", default=True, type=bool)
@click.option("--keras-autolog", help="Automatically log params, metrics and model with mlflow.keras.autolog", default=False, type=bool)
@click.option("--tensorflow-autolog", help="Automatically log params, metrics and model with mlflow.tensorflow.autolog", default=False, type=bool)
@click.option("--mlflow-autolog", help="Automatically log params, metrics and model with mlflow.autolog", default=False, type=bool)
@click.option("--log-as-onnx", help="log_as_onnx", default=False, type=bool)

def main(experiment_name, model_name, data_path, epochs, batch_size, mlflow_autolog, keras_autolog, tensorflow_autolog, mlflow_custom_log, log_as_onnx):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    model_name = None if not model_name or model_name == "None" else model_name
    if not mlflow_autolog and not keras_autolog and not tensorflow_autolog:
        mlflow_custom_log = True

    if keras_autolog:
        mlflow.keras.autolog()
    if tensorflow_autolog:
        mlflow.tensorflow.autolog()
    if mlflow_autolog:
        mlflow.autolog()

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        print("MLflow:")
        print("  run_id:",run.info.run_id)
        print("  experiment_id:",run.info.experiment_id)
        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.keras", keras.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)
        mlflow.set_tag("mlflow_autolog", mlflow_autolog)
        mlflow.set_tag("tensorflow_autolog", tensorflow_autolog)
        mlflow.set_tag("keras_autolog", keras_autolog)
        mlflow.set_tag("mlflow_custom_log", mlflow_custom_log)
        train(run, model_name, data_path, epochs, batch_size, mlflow_custom_log, log_as_onnx)

if __name__ == "__main__":
    main()
