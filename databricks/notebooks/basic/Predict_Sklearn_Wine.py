# Databricks notebook source
# MAGIC %md ## Sklearn model predict only
# MAGIC
# MAGIC ##### Overview
# MAGIC * Predicts using Sklearn, Pyfunc and Spark UDF flavors
# MAGIC * See notebook [Sklearn_Wine]($Sklearn_Wine)
# MAGIC
# MAGIC ##### Widgets
# MAGIC * `Model URI` - `models:/Sklearn_Wine/5` or `runs:/2620b314f33449078a6cb1a770a82de2/model`
# MAGIC * `Table` - if not specified, will load data from default sklearn location

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Model URI", "models:/andre_catalog.ml_models2.sklearn_wine_best@champ")
model_uri = dbutils.widgets.get("1. Model URI")

dbutils.widgets.text("2. Table", "andre_catalog.ml_data.winequality_white")
table_name = dbutils.widgets.get("2. Table")

print("model_uri:", model_uri)
print("table_name:", table_name)

toggle_unity_catalog(model_uri)

assert_widget(model_uri, "Model URI")

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

if table_name:
    data = spark.table(table_name).toPandas()
else:
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
