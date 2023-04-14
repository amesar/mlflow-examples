# Databricks notebook source
# MAGIC %md # Basic Keras MNIST train and predict notebook
# MAGIC * Trains and saves model as Keras flavor which uses the TensorFlow SavedModel format.
# MAGIC * Predicts using Keras model.
# MAGIC 
# MAGIC Widgets:
# MAGIC * Registered Model - If set, register the model under this name.
# MAGIC * Epochs - Number of epochs.
# MAGIC * Batch Size - Batch size.
# MAGIC 
# MAGIC Last update: 2023-02-20

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Registered Model", "") 
registered_model = dbutils.widgets.get("1. Registered Model")

dbutils.widgets.text("2. Epochs", "2") 
dbutils.widgets.text("3. Batch Size", "128")

epochs = int(dbutils.widgets.get("2. Epochs"))
batch_size = int(dbutils.widgets.get("3. Batch Size"))
if registered_model.strip() == "": registered_model = None

epochs, batch_size, registered_model
print("registered_model:", registered_model)
print("epochs:", epochs)
print("batch_size:", batch_size)

# COMMAND ----------

import tensorflow.keras as keras
import tensorflow as tf
import mlflow
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

import mlflow.keras

with mlflow.start_run(run_name=f"{now} - {mlflow.__version__}") as run:
    print("Run ID:", run.info.run_id)
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
    
    mlflow.log_param("my_epochs",epochs)
    mlflow.log_param("my_batch_size",batch_size)
    mlflow.log_metric("my_acc", test_acc)
    mlflow.log_metric("my_loss", test_loss)
    mlflow.keras.log_model(model, "model", registered_model_name=registered_model) 
    with open("/tmp/model.json", "w") as f:
        f.write(model.to_json())
    mlflow.log_artifact("/tmp/model.json")

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

# MAGIC %md ### Display links

# COMMAND ----------

display_run_uri(run.info.experiment_id, run.info.run_id)

# COMMAND ----------

if registered_model:
    display_registered_model_uri(registered_model)

# COMMAND ----------

# MAGIC %md ### Predict

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
