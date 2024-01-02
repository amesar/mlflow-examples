# Databricks notebook source
# MAGIC %md # Train model and predict best model with AutoML API
# MAGIC
# MAGIC ##### Overview
# MAGIC * Trains a model using the Databricks AutoML API feature.
# MAGIC * Runs will be in the notebook experiment.
# MAGIC * Uses the regression algorithm.
# MAGIC * Uses the wine quality dataset.
# MAGIC
# MAGIC ##### Widgets
# MAGIC * 1\. Table - Delta table. If not specified will read from CSV file.
# MAGIC * 2\. Timeout minutes - Maximum time to wait for AutoML trials to complete.
# MAGIC * 3\. Primary metric - Metric used to evaluate and rank model performance. 
# MAGIC   * Supported metrics for regression: “r2” (default), “mae”, “rmse”, “mse”.
# MAGIC * 4\. Best primary metric sort order - Sort order for searching for the best primary metric. Values are ASC or DESC (default).
# MAGIC * See [Classification and regression parameters](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/automl#classification-and-regression) documentation.
# MAGIC
# MAGIC ##### Databricks Documentation
# MAGIC * [Train ML models with Databricks AutoML Python API](https://docs.databricks.com/en/machine-learning/automl/train-ml-model-automl-api.html)
# MAGIC
# MAGIC ##### Github
# MAGIC * https://github.com/amesar/mlflow-examples/blob/master/databricks/notebooks/automl_api/AutoML_API_Example.py

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %md #### Include utilities

# COMMAND ----------

# MAGIC %run ./Utils

# COMMAND ----------

# MAGIC %md #### Create widgets

# COMMAND ----------

dbutils.widgets.text("1. Delta table", "andre_catalog.ml_data.winequality_white")
table_name = dbutils.widgets.get("1. Delta table")

dbutils.widgets.text("2. Timeout minutes", "5")
timeout_minutes = int(dbutils.widgets.get("2. Timeout minutes"))

dbutils.widgets.dropdown("3. Primary metric", "r2", ["r2", "mae", "rmse", "mse"])
primary_metric = dbutils.widgets.get("3. Primary metric") 

dbutils.widgets.dropdown("4. Best primary metric sort order","DESC",["ASC","DESC"])
best_primary_metric_sort_order = dbutils.widgets.get("4. Best primary metric sort order")

print("table_name:", table_name)
print("timeout_minutes:", timeout_minutes)
print("primary_metric:", primary_metric)
print("best_primary_metric_sort_order:", best_primary_metric_sort_order)

# COMMAND ----------

# MAGIC %md ### Get Data

# COMMAND ----------

if table_name:
    data = spark.table(table_name)
else:
    import pandas as pd
    path = "https://raw.githubusercontent.com/amesar/mlflow-examples/master/data/train/wine-quality-white.csv"
    data = pd.read_csv(path)
display(data)

# COMMAND ----------

# MAGIC %md ### Train model

# COMMAND ----------

from databricks.automl import regress
res = regress(
    dataset = data,
    target_col = "quality",
    primary_metric = primary_metric,
    timeout_minutes = timeout_minutes,
)

# COMMAND ----------

# MAGIC %md ### Display training results

# COMMAND ----------

experiment = res.experiment
display_experiment_uri(experiment)

# COMMAND ----------

print("Experiment ID:",experiment.experiment_id)
print("Experiment name:",experiment.name)

# COMMAND ----------

for k,v in res.__dict__.items():
    print("====",k[1:])
    print(f"\n{v}\n")

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

# MAGIC %md #### Find best run for the primary metric

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()
if primary_metric == "r2":
    primary_metric = f"{primary_metric}_score"
primary_metric

# COMMAND ----------

runs = client.search_runs(experiment.experiment_id, order_by=[f"metrics.test_{primary_metric} {best_primary_metric_sort_order}"], max_results=1)
best_run = runs[0]
print("run_id:",best_run.info.run_id)
print(f"primary_metric '{primary_metric}':",best_run.data.metrics[f"test_{primary_metric}"])

# COMMAND ----------

best_run.data.metrics

# COMMAND ----------

display_run_uri(experiment, best_run)

# COMMAND ----------

# MAGIC %md #### Create model URI

# COMMAND ----------

model_uri = f"runs:/{best_run.info.run_id}/model"
model_uri

# COMMAND ----------

# MAGIC %md #### Prepare prediction data

# COMMAND ----------

data_to_predict = data.drop("quality", axis=1)

# COMMAND ----------

# MAGIC %md #### Predict as PyFunc - scoring only on cluster driver

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
df = pd.DataFrame(predictions,columns=["prediction"])
df = df.round({'prediction': 3})
display(df)

# COMMAND ----------

# MAGIC %md #### Predict as Spark UDF - distributed scoring on cluster workers

# COMMAND ----------

from pyspark.sql.functions import *
df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns)).select("prediction")
predictions = predictions.withColumn("prediction", round(col("prediction"), 3))
display(predictions)
