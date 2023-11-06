# Databricks notebook source
# MAGIC %md ## Sklearn model predict only
# MAGIC
# MAGIC ##### Overview
# MAGIC * Predicts using Sklearn, Pyfunc and Spark UDF flavors
# MAGIC * See notebook [Sklearn_Wine]($Sklearn_Wine)
# MAGIC
# MAGIC ##### Widgets
# MAGIC * `Model URI` - `models:/Sklearn_Wine/5` or `runs:/2620b314f33449078a6cb1a770a82de2/model`

# COMMAND ----------

# MAGIC %md ### Setup

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

# MAGIC %md ### Predict as Sklearn flavor

# COMMAND ----------

import mlflow

model = mlflow.sklearn.load_model(model_uri)
predictions = model.predict(data_to_predict)

type(predictions), predictions.shape

# COMMAND ----------

display(pd.DataFrame(predictions, columns=[WineQuality.colPrediction]))

# COMMAND ----------

# MAGIC %md ### Predict as Pyfunc flavor

# COMMAND ----------

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
