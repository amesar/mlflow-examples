# Databricks notebook source
# MAGIC %md #Batch Scoring
# MAGIC * Scores the best model run from the [01_Train_Model]($01_Train_Model) notebook.
# MAGIC * Uses the model URI: `models:/mini_mlops_pipeline/production`.
# MAGIC * Scores with native Sklearn, Pyfunc and UDF flavors.
# MAGIC * Sklearn and Pyfunc scoring is executed only on the driver node, whereas UDF scoring leverages all nodes of the cluster.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

# MAGIC %md ### Prepare scoring data
# MAGIC * Drop the label column.

# COMMAND ----------

data = get_wine_quality_data()
data = data.drop([_col_label], axis=1)

# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md ### Build model URI using model alias `@a`

# COMMAND ----------

_model_name

# COMMAND ----------

model_uri = f"models:/{_model_name}@production"
model_uri

# COMMAND ----------

# MAGIC %md ### Score with native Sklearn flavor
# MAGIC * Executes only on the driver node of the cluster.

# COMMAND ----------

import pandas as pd
import mlflow.sklearn

model = mlflow.sklearn.load_model(model_uri)
predictions = model.predict(data)
display(pd.DataFrame(predictions, columns=[_col_prediction]))

# COMMAND ----------

# MAGIC %md ### Score with Pyfunc flavor
# MAGIC * Executes only on the driver node of the cluster.

# COMMAND ----------

import mlflow.pyfunc

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data)
display(pd.DataFrame(predictions, columns=[_col_prediction]))

# COMMAND ----------

# MAGIC %md ### Distributed scoring with UDF
# MAGIC * Executes on all worker nodes of the cluster.
# MAGIC * UDF wraps the Sklearn model.
# MAGIC * Pass a Spark dataframe to the UDF.
# MAGIC * The dataframe is split into multiple pieces and sent to each worker in the cluster for scoring.

# COMMAND ----------

df = spark.createDataFrame(data)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn(_col_prediction, udf(*df.columns))
display(predictions.select(_col_prediction))

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC Optionally, go to the **[04a_RT_Serving_Start]($04a_RT_Serving_Start)** notebook for real-time scoring.
