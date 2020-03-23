# Databricks notebook source
# MAGIC %md # Register and score model with Model Registry
# MAGIC * Scores the best model run from the [01_MLflow_SparkML_Tutorial]($01_MLflow_SparkML_Tutorial) notebooks.
# MAGIC * Creates a new registered model if it doesn't already exist
# MAGIC * Deletes all model versions
# MAGIC * Adds the best run as a `production` version.
# MAGIC * Loads the model as `models:/Tutorial_Model/production`
# MAGIC * Scores with Pyfunc flavor.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("Registered model name", "Tutorial_Model")
model_name = dbutils.widgets.get("Registered model name")

# COMMAND ----------

import os
notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
dir = os.path.dirname(notebook)
experiment_name = f"{dir}/01_MLflow_SparkML_Tutorial"
experiment_name

# COMMAND ----------

# MAGIC %md ### MLflow Setup

# COMMAND ----------

import mlflow
print("MLflow Version:", mlflow.__version__)
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id
display_experiment_uri(experiment_id)

# COMMAND ----------

# MAGIC %md ### Find best run

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
best_run = runs[0]
best_run.info.run_id, best_run.data.metrics["rmse"], best_run.data.params

# COMMAND ----------

# MAGIC %md ### Wait functions due to eventual consistency
# MAGIC * Wait until a version is in the READY status

# COMMAND ----------

import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_version_ready(model_name, model_version, sleep_time=1, iterations=100):
    start = time.time()
    for _ in range(iterations):
        version = client.get_model_version(model_name, model_version.version)
        status = ModelVersionStatus.from_string(version.status)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(round(time.time())))
        print(f"{dt}: Version {version.version} status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(sleep_time)
    end = time.time()
    print(f"Waited {round(end-start,2)} seconds")

# COMMAND ----------

# MAGIC %md ### Create registered model (if it doen't exist) and remove all versions

# COMMAND ----------

from mlflow.exceptions import MlflowException, RestException

try:
    registered_model = client.get_registered_model(model_name)
    print(f"Found {model_name}")
    versions = client.get_latest_versions(model_name)
    print(f"Found {len(versions)} versions")
    for v in versions:
        print(f"  version={v.version} status={v.status} stage={v.current_stage} run_id={v.run_id}")
        client.update_model_version(model_name, v.version, stage="Archived", description="to archived")
        client.delete_model_version(model_name, v.version)
except RestException as e:
    print("INFO:",e)
    if e.error_code == "RESOURCE_DOES_NOT_EXIST":
        print(f"Creating {model_name}")
        registered_model = client.create_registered_model(model_name)
    else:
        raise Exception(e)

# COMMAND ----------

registered_model = client.get_registered_model(model_name)
type(registered_model),registered_model.__dict__

# COMMAND ----------

display_registered_model_uri(model_name)

# COMMAND ----------

# MAGIC %md ### Create model version for run

# COMMAND ----------

# MAGIC %md **Create the version**

# COMMAND ----------

source = f"{best_run.info.artifact_uri}/spark-model"
source

# COMMAND ----------

version = client.create_model_version(model_name, source, best_run.info.run_id)
version

# COMMAND ----------

# MAGIC %md **Wait until version is in READY status**

# COMMAND ----------

wait_until_version_ready(model_name, version, sleep_time=2)
version = client.get_model_version(model_name,version.version)
version_id = version.version
version_id, version.status, version.current_stage, version.run_id

# COMMAND ----------

# MAGIC %md ### Tag best run version as Production stage

# COMMAND ----------

client.update_model_version(model_name, version_id, stage="Production", description="My prod version")

# COMMAND ----------

version = client.get_model_version(model_name,version_id)
type(version), version.__dict__

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

import pandas as pd
data = pd.read_csv(download_data())
data = data.drop([colLabel], axis=1)
display(data)

# COMMAND ----------

# MAGIC %md ### Score with pyfunc flavor

# COMMAND ----------

# MAGIC %md #### Load model with `models` URI
# MAGIC * Note we use`models` scheme instead of `runs` scheme.
# MAGIC * This way you can update the production run without impacting downstream scoring code

# COMMAND ----------

model_uri = f"models:/{model_name}/production"
model_uri

# COMMAND ----------

import mlflow.pyfunc
model = mlflow.pyfunc.load_model(model_uri)
type(model)

# COMMAND ----------

# MAGIC %md #### Score data

# COMMAND ----------

predictions = model.predict(data)
type(predictions)

# COMMAND ----------

pd.DataFrame(predictions).head(10)