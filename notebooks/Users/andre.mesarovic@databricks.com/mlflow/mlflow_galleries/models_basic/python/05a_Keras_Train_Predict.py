# Databricks notebook source
# MAGIC %md # Basic Keras 2.x MNIST train and predict notebook
# MAGIC * Trains and saves model as Keras flavor which uses the TensorFlow SavedModel format.
# MAGIC * Predicts using Keras model.
# MAGIC 
# MAGIC Widgets:
# MAGIC * Autolog:
# MAGIC   * None - no autologging - explicitly log the model, params and metrics.
# MAGIC   * mlflow - call mlflow.autolog().
# MAGIC   * tensoflow - call mlflow.tensorflow.autolog().
# MAGIC   * keras - call mlflow.keras.autolog() - as of DBR ML 10.3 works fine. Previously croaked with `ModuleNotFoundError: No module named 'keras'`.
# MAGIC * Epochs - Number of epochs.
# MAGIC * Batch Size - Batch size.
# MAGIC * Registered Model - If set, register the model under this name.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("Epochs", "2") 
dbutils.widgets.text("Batch Size", "128")
dbutils.widgets.text("Registered Model","")
dbutils.widgets.dropdown("Autolog","None",["None","mlflow","tensorflow","keras"])

epochs = int(dbutils.widgets.get("Epochs"))
batch_size = int(dbutils.widgets.get("Batch Size"))
autolog = dbutils.widgets.get("Autolog")
registered_model = dbutils.widgets.get("Registered Model")
if registered_model.strip() == "": registered_model = None

epochs, batch_size, autolog, registered_model

# COMMAND ----------

import tensorflow.keras as keras
import tensorflow as tf
import mlflow
import mlflow.keras

if autolog == "mlflow":
    mlflow.autolog()
elif autolog == "tensorflow":
    mlflow.tensorflow.autolog()
elif autolog == "keras":
    mlflow.keras.autolog() # ERROR: ModuleNotFoundError: No module named 'keras'

# COMMAND ----------

import numpy as np

np.random.seed(1)
tf.random.set_seed(1)

# COMMAND ----------

print("sparkVersion:", get_notebook_tag("sparkVersion"))
print("MLflow Version:", mlflow.__version__)
print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

# COMMAND ----------

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("K.image_data_format:",K.image_data_format())
print("train_images.shape:",train_images.shape)
print("train_labels.shape:",train_labels.shape)
print("train_labels:",train_labels)
print("test_images.shape:",test_images.shape)
print("test_labels.shape:",test_labels.shape)
print("test_labels:",test_labels)

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255 

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255 

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# COMMAND ----------

input_shape = (28 * 28,)
print("input_shape:",input_shape)

# COMMAND ----------

print("train_images.shape:",train_images.shape)
print("train_labels.shape:",train_labels.shape)
print("test_images.shape:",test_images.shape)
print("test_labels.shape:",test_labels.shape)

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

model = models.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=input_shape))
model.add(layers.Dense(10, activation='softmax'))

# COMMAND ----------

with mlflow.start_run() as run:
    print("Run ID:", run.info.run_id)
    mlflow.set_tag("autolog", autolog)
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("version.tensorflow", tf.__version__)
    mlflow.set_tag("version.keras", keras.__version__)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_acc:", test_acc)
    print("test_loss:", test_loss)
    
    if autolog == "None":
        mlflow.log_param("my_epochs",epochs)
        mlflow.log_param("my_batch_size",batch_size)
        mlflow.log_metric("my_acc", test_acc)
        mlflow.log_metric("my_loss", test_loss)
        mlflow.keras.log_model(model, "model", registered_model_name=registered_model) 
        with open("model.json", "w") as f:
            f.write(model.to_json())
        mlflow.log_artifact("model.json")

# COMMAND ----------

test_images.shape

# COMMAND ----------

# AttributeError: 'PyFuncModel' object has no attribute 'predict_classes'
#predictions = model.predict_classes(test_images) # Error: 'Sequential' object has no attribute 'predict_classes'
#predictions

# COMMAND ----------

predictions = model.predict(test_images)
pd.DataFrame(data=predictions)

# COMMAND ----------

display_run_uri(run.info.experiment_id, run.info.run_id)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

autolog

# COMMAND ----------

model_uri = f"runs:/{run.info.run_id}/model"
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as Keras

# COMMAND ----------

test_images.shape

# COMMAND ----------

model = mlflow.keras.load_model(model_uri)
predictions = model.predict(test_images)
pd.DataFrame(data=predictions)

# COMMAND ----------

type(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as Pyfunc

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
model

# COMMAND ----------

test_images_pd = pd.DataFrame(data=test_images)
test_images_pd.shape

# COMMAND ----------

predictions = model.predict(test_images_pd)
pd.DataFrame(data=predictions)

# COMMAND ----------

type(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as UDF - TODO
# MAGIC 
# MAGIC * TypeError: Can not infer schema for type: ``<class 'numpy.ndarray'>``

# COMMAND ----------

#df = spark.createDataFrame(test_images)
#udf = mlflow.pyfunc.spark_udf(spark, model_uri)
#predictions = df.withColumn("prediction", udf(*df.columns))
#predictions