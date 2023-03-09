# Databricks notebook source
# MAGIC %md # Register best model in Model Registry
# MAGIC 
# MAGIC * Finds the best run (lowest RMSE metric). 
# MAGIC * Creates a new registered model if it doesn't already exist.
# MAGIC * Deletes all previous model versions (only for demo purposes so we start with a clean slate).
# MAGIC * Adds the best run's model to the Model Registry as a new version and promotes it to the `production` stage.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------


dbutils.widgets.text("Registered model name", default_model_name)
model_name = dbutils.widgets.get("Registered model name")
model_name

# COMMAND ----------

#import os
#notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
#notebook_dir = os.path.dirname(notebook_path)
#experiment_name = os.path.join(notebook_dir, experiment_name)
#experiment_name

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

# MAGIC %md ### Find best run with the lowest RMSE metric

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["metrics.training_rmse ASC"], max_results=1)
best_run = runs[0]
best_run.info.run_id, best_run.data.metrics["training_rmse"], best_run.data.params

# COMMAND ----------

# MAGIC %md ### Create registered model (if it doen't exist) 
# MAGIC * Remove all previous versions so to have a clean slate for demo purposes.
# MAGIC * In a real world case, you probably wouldn't do this.

# COMMAND ----------

from mlflow.exceptions import MlflowException, RestException

try:
    registered_model = client.get_registered_model(model_name)
    print(f"Found {model_name}")
    versions = client.get_latest_versions(model_name)
    print(f"Found {len(versions)} versions")
    for v in versions:
        print(f"  version={v.version} status={v.status} stage={v.current_stage} run_id={v.run_id}")
        client.transition_model_version_stage(model_name, v.version, "Archived")
        client.delete_model_version(model_name, v.version) # only for demo purposes
except RestException as e:
    print("INFO:",e)
    if e.error_code == "RESOURCE_DOES_NOT_EXIST":
        print(f"Creating {model_name}")
        registered_model = client.create_registered_model(model_name)
    else:
        raise Exception(e)

# COMMAND ----------

registered_model = client.get_registered_model(model_name)
type(registered_model), registered_model.__dict__

# COMMAND ----------

display_registered_model_uri(model_name)

# COMMAND ----------

# MAGIC %md ### Create model version for run

# COMMAND ----------

# MAGIC %md **Create the version**

# COMMAND ----------

source = f"{best_run.info.artifact_uri}/model"
source

# COMMAND ----------

version = client.create_model_version(model_name, source, best_run.info.run_id)
version.__dict__

# COMMAND ----------

# MAGIC %md **Wait until version is in READY status**

# COMMAND ----------

wait_until_version_ready(model_name, version, sleep_time=2)
version = client.get_model_version(model_name,version.version)
version_id = version.version
version_id, version.status, version.current_stage, version.run_id

# COMMAND ----------

# MAGIC %md ### Promote best run version to Production stage

# COMMAND ----------

version = client.transition_model_version_stage(model_name, version_id, stage="Production")
version.__dict__