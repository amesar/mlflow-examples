# Databricks notebook source
# MAGIC %md # Search MLflow runs
# MAGIC 
# MAGIC **Overview**
# MAGIC * Shows different ways to view and search runs:
# MAGIC   * [MlflowClient.list_run_infos](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.list_run_infos) - returns list of [RunInfo](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo) objects
# MAGIC   * [MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs) - returns list of [Run](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run) objects.
# MAGIC   * [mlflow.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs) - returns Pandas DataFrame that represents all runs.
# MAGIC   * [Experiment data source](https://docs.databricks.com/applications/mlflow/tracking.html#analyze-mlflow-runs-using-dataframes) - returns a Spark DataFrame that represents all runs.
# MAGIC * Find the best run for the experiment using MlflowClient.search_runs.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md ### Show different ways to view an experiment's model runs

# COMMAND ----------

# MAGIC %md #### MlflowClient.list_run_infos
# MAGIC * [mlflow.tracking.MlflowClient.list_run_infos](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.list_run_infos)
# MAGIC * Returns a list of [RunInfo](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo) objects

# COMMAND ----------

infos = client.list_run_infos(experiment_id)
for info in infos:
    print(info.run_id, info.experiment_id, info.status)

# COMMAND ----------

# MAGIC %md #### MLflowClient.search_runs
# MAGIC * [mlflow.tracking.MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs)
# MAGIC * Returns a list of [Run](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run) objects,
# MAGIC * Allows for paging when you have a very large number of runs.
# MAGIC * Sorted by best metrics `rmse`.

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["metrics.training_rmse ASC"])
for run in runs:
    print(run.info.run_id, run.data.metrics["training_rmse"], run.data.params)

# COMMAND ----------

# MAGIC %md #### mlflow.search_runs
# MAGIC * [mlflow.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs)
# MAGIC * Returns a Pandas dataframe.
# MAGIC * All `data` attributes are exploded into one flat column name space.
# MAGIC * Sorted by best metrics `rmse`.

# COMMAND ----------

runs = mlflow.search_runs(experiment_id)
runs = runs.sort_values(by=['metrics.training_rmse'])
runs

# COMMAND ----------

runs[["run_id","metrics.training_rmse","params.max_depth"]]

# COMMAND ----------

# MAGIC %md #### Experiment data source
# MAGIC * Returns a Spark dataframe of all runs.
# MAGIC * Run `data` elements such as `params`, `metrics` and `tags` are nested.
# MAGIC * Background documentation:
# MAGIC   * Databricks documentation:
# MAGIC     * [MLflow Experiment Data Source](https://docs.databricks.com/data/data-sources/mlflow-experiment.html#mlflow-exp-datasource)
# MAGIC     * [Analyze MLflow runs using DataFrames
# MAGIC ](https://docs.databricks.com/applications/mlflow/tracking.html#analyze-mlflow-runs-using-dataframes)
# MAGIC   * [Analyzing Your MLflow Data with DataFrames](https://databricks.com/blog/2019/10/03/analyzing-your-mlflow-data-with-dataframes.html) - blog - 2019-10-03

# COMMAND ----------

from pyspark.sql.functions import *
df_runs = spark.read.format("mlflow-experiment").load(experiment_id)
df_runs.createOrReplaceTempView("runs")

# COMMAND ----------

# MAGIC %md ##### Query with Spark DataFrame API

# COMMAND ----------

df_runs = df_runs.sort(asc("metrics.training_rmse"))
display(df_runs)

# COMMAND ----------

display(df_runs.select("run_id", round("metrics.training_rmse",3).alias("training_rmse"),"params"))

# COMMAND ----------

# MAGIC %md ##### Query as SQL

# COMMAND ----------

# MAGIC %sql select run_id, metrics.rmse, params from runs order by metrics.rmse asc

# COMMAND ----------

# MAGIC %sql select run_id, metrics.rmse, params from runs order by metrics.rmse asc limit 1

# COMMAND ----------

# MAGIC %md ### Find the best run
# MAGIC 
# MAGIC * We use `MlflowClient.search_run` to find the best run

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
best_run = runs[0]
best_run

# COMMAND ----------

print("Run ID:",best_run.info.run_id)
print("RMSE:",best_run.data.metrics["training_rmse"])

# COMMAND ----------

display_run_uri(experiment_id, best_run.info.run_id)