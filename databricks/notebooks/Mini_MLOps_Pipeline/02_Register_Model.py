# Databricks notebook source
# MAGIC %md # Register Best Model in Model Registry
# MAGIC * Creates a new registered model if it doesn't already exist.
# MAGIC * Deletes all current model versions (optional).
# MAGIC * Finds the best model (lowest 'training_rmse' metric) generated from [01_Train_Model]($01_Train_Model) notebook experiment.
# MAGIC * Use 'training_mean_absolute_error' if you autolog your run.
# MAGIC * Adds the best run as a registered model version and promotes it to the `production` stage.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

dbutils.widgets.text("1. Metric", "training_rmse")
metric = dbutils.widgets.get("1. Metric")

dbutils.widgets.dropdown("2. Delete registered model", "yes", ["yes", "no"])
delete_registered_model = dbutils.widgets.get("2. Delete registered model") == "yes"

print("metric:", metric)
print("delete_registered_model:", delete_registered_model)

# COMMAND ----------

experiment = mlflow_client.get_experiment_by_name(_experiment_name)
experiment_id = experiment.experiment_id
display_experiment_uri(experiment)

# COMMAND ----------

# MAGIC %md ### Show all the RMSE metrics and max_depth params of all runs

# COMMAND ----------

df = mlflow.search_runs(experiment_id, order_by=[f"metrics.{metric} ASC"])
df = df[["run_id",f"metrics.{metric}","params.max_depth"]]
df = df.round({f"metrics.{metric}": 3})
display(df)

# COMMAND ----------

# MAGIC %md ### Find best run
# MAGIC
# MAGIC Search for the run with lowest training RMSE metric.

# COMMAND ----------

runs = mlflow_client.search_runs(experiment_id, order_by=[f"metrics.{metric} ASC"], max_results=1)
best_run = runs[0]
best_run.info.run_id, round(best_run.data.metrics[metric],3), best_run.data.params

# COMMAND ----------

# MAGIC %md ##### If you used a Delta table to train your model, its value is displayed here as in:
# MAGIC
# MAGIC `path=dbfs:/user/hive/warehouse/andre.db/wine_quality,version=0,format=delta`

# COMMAND ----------

best_run.data.tags.get("sparkDatasourceInfo")

# COMMAND ----------

# MAGIC %md ### Create registered model 
# MAGIC
# MAGIC If the model already exists remove, then delete all existing versions.

# COMMAND ----------

from mlflow.exceptions import MlflowException, RestException

try:
    registered_model = mlflow_client.get_registered_model(_model_name)
    print(f"Found registered model '{_model_name}'")
    if delete_registered_model:
        versions = mlflow_client.get_latest_versions(_model_name)
        print(f"Found {len(versions)} model versions to delete")
        for v in versions:
            print(f"  Deleting version={v.version} status={v.status} stage={v.current_stage} run_id={v.run_id}")
            mlflow_client.transition_model_version_stage(_model_name, v.version, "Archived")
            mlflow_client.delete_model_version(_model_name, v.version)
        print(f"Deleting registered model '{_model_name}'")
        mlflow_client.delete_registered_model(_model_name)
        registered_model = mlflow_client.create_registered_model(_model_name)
except RestException as e:
    if e.error_code == "RESOURCE_DOES_NOT_EXIST":
        print(f"Creating registered model '{_model_name}'")
        registered_model = mlflow_client.create_registered_model(_model_name)
    else:
        raise e

# COMMAND ----------

registered_model = mlflow_client.get_registered_model(_model_name)
type(registered_model),registered_model.__dict__

# COMMAND ----------

display_registered_model_uri(_model_name)

# COMMAND ----------

# MAGIC %md ### Create model version for best run

# COMMAND ----------

source = f"{best_run.info.artifact_uri}/model"
source

# COMMAND ----------

version = mlflow_client.create_model_version(
    name = _model_name, 
    source = source, 
    run_id = best_run.info.run_id
)
type(version), version.__dict__

# COMMAND ----------

# MAGIC %md ### Promote best run version to Production stage

# COMMAND ----------

# MAGIC %md **Wait until version is in READY status**

# COMMAND ----------

wait_until_version_ready(_model_name, version, sleep_time=2)

# COMMAND ----------

version = mlflow_client.get_model_version(_model_name, version.version)
version_id = version.version

print("version.version:", version.version)
print("version.status:", version.status)
print("version.current_stage:", version.current_stage)
print("version.run_id:", version.run_id)

# COMMAND ----------

# MAGIC %md **Transition to production stage**

# COMMAND ----------

mlflow_client.transition_model_version_stage(_model_name, version_id, "Production")

# COMMAND ----------

version = mlflow_client.get_model_version(_model_name, version_id)
type(version), version.__dict__

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC Now either go to:
# MAGIC * **[03a_Batch_Scoring]($03a_Batch_Scoring)** notebook for batch scoring
# MAGIC * **[04a_RT_Serving_Start ]($04a_RT_Serving_Start )** notebook for real-time model serving
