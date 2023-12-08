# Databricks notebook source
# MAGIC %md ## Batch Scoring
# MAGIC * Scores the best model run from the [01_Train_Model]($01_Train_Model) notebook.
# MAGIC * Scores with native Sklearn, Pyfunc and UDF flavors.
# MAGIC * Shows how to load a model using either the standard version or new alias.
# MAGIC * Sklearn and Pyfunc scoring is executed only on the driver node, whereas UDF scoring uses all nodes of the cluster.
# MAGIC
# MAGIC ##### Widgets
# MAGIC
# MAGIC * `1. Registered model` - name of registerd model such as `andre_m.ml_models.diabetes_mlops`.
# MAGIC * `2. Table` - diabetes table such as `andre_m.ml_data.diabetes` (optional).
# MAGIC * `3. Alias` - Model alias.

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
model_name = dbutils.widgets.get("1. Registered model")
assert_widget(model_name, "1. Registered model")

dbutils.widgets.text("2. Table", "")
table_name = dbutils.widgets.get("2. Table")
table_name = table_name or None

dbutils.widgets.text("3. Alias", _alias)
alias = dbutils.widgets.get("3. Alias")

print("model_name:", model_name)
print("table_name:", table_name)
print("alias:", alias)

# COMMAND ----------

# MAGIC %md #### Prepare scoring data
# MAGIC * Drop the label column `progression`

# COMMAND ----------

data = load_data(table_name)
data = data.drop(["progression"], axis=1)

# COMMAND ----------

# MAGIC %md #### Prepare model URI
# MAGIC * A `model URI` can use either a model version's version or alias.
# MAGIC   * With version number: `models:/my_catalog.models.diabetes_mlops/1`
# MAGIC   * With alias: `models:/my_catalog.models.diabetes_mlops@alias`

# COMMAND ----------

if alias:
    model_uri = f"models:/{model_name}@{alias}"
else:
    model_uri = f"models:/{model_name}/1"
model_uri

# COMMAND ----------

# MAGIC
# MAGIC %md #### Score with native Sklearn flavor
# MAGIC * Executes only on the driver node

# COMMAND ----------

import pandas as pd

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
predictions = model.predict(data)
display(pd.DataFrame(predictions, columns=["prediction"]))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #### Score with Pyfunc flavor
# MAGIC * Executes only on the driver node of the cluster

# COMMAND ----------

import mlflow.pyfunc

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data)
display(pd.DataFrame(predictions, columns=["prediction"]))

# COMMAND ----------

# MAGIC %md #### Distributed scoring with UDF
# MAGIC * Executes on all worker nodes of the cluster.
# MAGIC * UDF wraps the Sklearn model.
# MAGIC * Pass a Spark dataframe to the UDF.
# MAGIC * The dataframe is split into multiple chunk and sent to each worker in the cluster for scoring.

# COMMAND ----------

df = spark.createDataFrame(data)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns))
display(predictions.select("prediction"))
