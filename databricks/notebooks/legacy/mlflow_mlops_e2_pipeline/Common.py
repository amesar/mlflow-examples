# Databricks notebook source
# Common utilities

# COMMAND ----------

def _get_notebook_tag(tag):
    tag = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get(tag)
    return None if tag.isEmpty() else tag.get()

# COMMAND ----------

import os
import platform
import mlflow
import mlflow.spark
import pyspark
print("MLflow Version:", mlflow.__version__)
print("Spark Version:", spark.version)
print("PySpark Version:", pyspark.__version__)
print("sparkVersion:", _get_notebook_tag("sparkVersion"))
print("DATABRICKS_RUNTIME_VERSION:", os.environ.get('DATABRICKS_RUNTIME_VERSION',None))
print("Python Version:", platform.python_version())

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

default_model_name = "MLflow MLOps E2E Pipeline"
print("default_model_name:",default_model_name)

notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
notebook_dir = os.path.dirname(notebook)
#notebook_name = "01_MLflow_Sklearn_Train"
notebook_name = "01_Train_Model"

experiment_name = os.path.join(notebook_dir, notebook_name)
print("Experiment name:",experiment_name)

experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id
print("Experiment ID:",experiment.experiment_id)

# COMMAND ----------

data_path = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
print("data_path:",data_path)

# COMMAND ----------

col_label = "quality"
col_prediction = "prediction"

# COMMAND ----------

host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()

# COMMAND ----------

def delete_runs(experiment_id):
    run_infos = client.list_run_infos(experiment_id)
    print(f"Found {len(run_infos)} runs for experiment_id {experiment_id}")
    for run_info in run_infos:
        client.delete_run(run_info.run_id)

# COMMAND ----------

def display_run_uri(experiment_id, run_id):
    uri = f"https://{host_name}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
    displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_experiment_uri(experiment_id):
    uri = "https://{}/#mlflow/experiments/{}".format(host_name, experiment_id)
    displayHTML("""<b>Experiment URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_registered_model_uri(model_name):
    uri = f"https://{host_name}/#mlflow/models/{model_name}"
    displayHTML("""<b>Registered Model URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

# Wait until a version is in the READY status.
# Needed due to cloud eventual consistency.

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