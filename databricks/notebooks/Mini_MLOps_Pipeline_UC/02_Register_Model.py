# Databricks notebook source
# MAGIC %md # Register Best Model in Model Registry
# MAGIC * Creates a new registered model if it doesn't already exist.
# MAGIC * Deletes all current model versions (optional).
# MAGIC * Finds the best model (lowest RMSE metric) generated from [01_Train_Model]($01_Train_Model) notebook experiment.
# MAGIC * Adds the best run as a registered model version and promotes it to the `production` stage.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model","")
_model_name = dbutils.widgets.get("1. Registered model") 
_write_model_name()

dbutils.widgets.dropdown("2. Delete existing versions","yes",["yes","no"])
delete_existing_versions = dbutils.widgets.get("2. Delete existing versions") == "yes"

print("_model_name:", _model_name)
print("delete_existing_versions:", delete_existing_versions)

# COMMAND ----------

assert_widget(_model_name, "1. Registered model")

# COMMAND ----------

# MAGIC %md ### Display experiment

# COMMAND ----------

experiment = mlflow_client.get_experiment_by_name(_experiment_name)
experiment_id = experiment.experiment_id
display_experiment_uri(experiment)

# COMMAND ----------

# MAGIC %md ### Show all the RMSE metrics and max_depth params of all runs

# COMMAND ----------

df = mlflow.search_runs(experiment_id, order_by=["metrics.rmse ASC"])
df = df[["run_id","metrics.training_rmse","params.max_depth"]]
df = df.round({"metrics.training_rmse": 3})
display(df)

# COMMAND ----------

# MAGIC %md ### Find best run
# MAGIC
# MAGIC Search for the run with lowest RMSE metric.

# COMMAND ----------

runs = mlflow_client.search_runs(experiment_id, order_by=["metrics.training_rmse ASC"], max_results=1)
best_run = runs[0]
best_run.info.run_id, round(best_run.data.metrics["training_rmse"],3), best_run.data.params

# COMMAND ----------

# MAGIC %md ##### If you used a Delta table to train your model, its value is displayed here as in:
# MAGIC
# MAGIC `path=dbfs:/user/hive/warehouse/andre.db/wine_quality,version=0,format=delta`

# COMMAND ----------

best_run.data.tags.get("sparkDatasourceInfo")

# COMMAND ----------

# MAGIC %md ### Create registered model 
# MAGIC
# MAGIC If model already exists remove all existing versions.

# COMMAND ----------

from mlflow.exceptions import MlflowException, RestException
from mlflow.exceptions import MlflowException, RestException

try:
    registered_model = mlflow_client.get_registered_model(_model_name)
    print(f"Found existing {_model_name}")
    versions = mlflow_client.search_model_versions(f"name='{_model_name}'")
    print(f"Found {len(versions)} versions")
    if delete_existing_versions:
        for vr in versions:
            print(f"  version={vr.version} status={vr.status} stage={vr.current_stage} run_id={vr.run_id}")
            mlflow_client.delete_model_version(_model_name, vr.version)
except RestException as e:
    print("INFO:",e)
    if e.error_code == "RESOURCE_DOES_NOT_EXIST":
        print(f"Creating {_model_name}")
        registered_model = mlflow_client.create_registered_model(_model_name)
    else:
        raise RuntimeError(e)

# COMMAND ----------

type(registered_model),registered_model.__dict__

# COMMAND ----------

display_registered_model_uri(_model_name)

# COMMAND ----------

# MAGIC %md **Create the version**

# COMMAND ----------

source = f"{best_run.info.artifact_uri}/model"
source

# COMMAND ----------

 mlflow_client.tracking_uri, mlflow_client._registry_uri, _model_name

# COMMAND ----------

version = mlflow_client.create_model_version(_model_name, source, best_run.info.run_id)
type(version), version.__dict__

# COMMAND ----------

# MAGIC %md ### Promote best run version to Production stage

# COMMAND ----------

# MAGIC %md **Wait until version is in READY status**

# COMMAND ----------

_model_name

# COMMAND ----------

wait_until_version_ready(_model_name, version, sleep_time=2)

# COMMAND ----------

version = mlflow_client.get_model_version(_model_name, version.version)
version_id = version.version

print("version.version:", version.version)
print("version.status:", version.status)
print("version.current_stage:", version.current_stage)
print("version.aliases:", version.aliases)
print("version.run_id:", version.run_id)

# COMMAND ----------

# MAGIC %md **Transition to production stage**

# COMMAND ----------

##mlflow_client.transition_model_version_stage(_model_name, version_id, "Production")
mlflow_client.set_registered_model_alias(_model_name, "production", version.version)

# COMMAND ----------

version = mlflow_client.get_model_version(_model_name, version_id)
type(version), version.__dict__

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC Now either go to:
# MAGIC * **[03a_Batch_Scoring]($03a_Batch_Scoring)** notebook for batch scoring
# MAGIC * **[04a_RT_Serving_Start ]($04a_RT_Serving_Start )** notebook for real-time model serving
