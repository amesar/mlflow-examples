# Databricks notebook source
# MAGIC %md ## ONNX model predict only
# MAGIC
# MAGIC ##### Overview
# MAGIC * Predicts using native ONNX, Pyfunc and Spark UDF flavors.
# MAGIC * See notebook [Sklearn_Wine_ONNX]($Sklearn_Wine_ONNX).
# MAGIC
# MAGIC ##### Widgets
# MAGIC * `Model URI` - `models:/Sklearn_Wine_ONNX/5` or `runs:/2620b314f33449078a6cb1a770a82de2/model`

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %pip install onnx==1.15.0
# MAGIC %pip install onnxruntime==1.16.1

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("Model URI", "")
model_uri = dbutils.widgets.get("Model URI")
print("model_uri:", model_uri)

assert_widget(model_uri, "Model URI")

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data = WineQuality.load_pandas_data()
data_to_predict = WineQuality.prep_prediction_data(data)
display(data_to_predict)

# COMMAND ----------

# MAGIC %md ### Predict as Pyfunc flavor

# COMMAND ----------

import mlflow

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)

type(predictions), predictions.shape

# COMMAND ----------

display(pd.DataFrame(predictions, columns=[WineQuality.colPrediction]))

# COMMAND ----------

# MAGIC %md ### Predict as Spark UDF

# COMMAND ----------

df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns)).select("prediction")
display(predictions)

# COMMAND ----------

# MAGIC %md ### Predict as native ONNX flavor

# COMMAND ----------

import mlflow.onnx
import onnxruntime
import numpy as np

model = mlflow.onnx.load_model(model_uri)
session = onnxruntime.InferenceSession(model.SerializeToString())
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name: data_to_predict.to_numpy().astype(np.float32)})[0]

type(predictions), predictions.shape

# COMMAND ----------

display(pd.DataFrame(predictions, columns=["prediction"]))
