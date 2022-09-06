# Databricks notebook source
# MAGIC %md # Batch score model from Model Registry
# MAGIC * Batch scoring.
# MAGIC * Loads the model as `models:/MLflow MLOps E2E Pipeline/production`.
# MAGIC * Show how to score the model with different flavors:
# MAGIC   * [Sklearn flavor](https://mlflow.org/docs/latest/models.html#scikit-learn-sklearn) 
# MAGIC   * [Pyfunc flavor](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) 
# MAGIC   * [UDF](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf)
# MAGIC * Distributed scoring
# MAGIC   * Scoring with the sklearn and pyfunc flavors will run on the driver node. 
# MAGIC   * Scoring with a UDF allows for distributed scoring across the cluster's worker nodes.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("Registered model name", default_model_name)
model_name = dbutils.widgets.get("Registered model name")
model_name

# COMMAND ----------

# MAGIC %md ### MLflow Setup

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

import pandas as pd
data = pd.read_csv(data_path)
data = data.drop([col_label], axis=1)
display(data)

# COMMAND ----------

# MAGIC %md ### Score
# MAGIC 
# MAGIC Several ways to score:
# MAGIC * Sklearn flavor
# MAGIC * PyFunc flavor
# MAGIC * Spark UDF flavor

# COMMAND ----------

# MAGIC %md #### Load model with `models` URI
# MAGIC * Note we use`models` scheme instead of `runs` scheme.
# MAGIC * This way you can update the production model without impacting downstream scoring code

# COMMAND ----------

model_uri = f"models:/{model_name}/production"
model_uri

# COMMAND ----------

# MAGIC %md #### Score with Sklearn flavor

# COMMAND ----------

import mlflow
model = mlflow.sklearn.load_model(model_uri)
type(model)

# COMMAND ----------

predictions = model.predict(data)
type(predictions)

# COMMAND ----------

display(pd.DataFrame(predictions,columns=[col_prediction]))

# COMMAND ----------

# MAGIC %md #### Score with PyFunc flavor

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
type(model)

# COMMAND ----------

predictions = model.predict(data)
type(predictions)

# COMMAND ----------

pd.DataFrame(predictions).head(10)

# COMMAND ----------

# MAGIC %md #### Score with Spark UDF

# COMMAND ----------

df = spark.createDataFrame(data)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns))
display(predictions.select("prediction"))