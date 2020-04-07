# Databricks notebook source
# Base directory is "/dbfs/users/${user}/tmp/mlflow_demo"

import os
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
base_dir_fuse = f"/dbfs/users/${user}/tmp/mlflow_demo"
os.makedirs(base_dir_fuse, exist_ok=True)
base_dir_fuse

# COMMAND ----------

import os
import mlflow
import mlflow.spark
import pyspark
print("MLflow Version:", mlflow.__version__)
print("Spark Version:", spark.version)
print("PySpark Version:", pyspark.__version__)
print("DATABRICKS_RUNTIME_VERSION:", os.environ.get('DATABRICKS_RUNTIME_VERSION',None))
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

def init():
    experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    print("Experiment name:",experiment_name)
    experiment = client.get_experiment_by_name(experiment_name)
    print("Experiment ID:",experiment.experiment_id)
    return experiment.experiment_id, experiment_name

# COMMAND ----------

def delete_runs(experiment_id):
    run_infos = client.list_run_infos(experiment_id)
    print(f"Found {len(run_infos)} runs for experiment_id {experiment_id}")
    for run_info in run_infos:
        client.delete_run(run_info.run_id)

# COMMAND ----------

import os
import requests

def download_file(data_uri, data_path):
    if os.path.exists(data_path):
        print(f"File {data_path} already exists")
    else:
        print(f"Downloading {data_uri} to {data_path}")
        rsp = requests.get(data_uri)
        with open(data_path, "w") as f:
            f.write(requests.get(data_uri).text)

def download_data():
    data_path = f"{base_dir_fuse}/mlflow_wine-quality.csv"
    data_uri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
    download_file(data_uri, data_path)
    return data_path

# COMMAND ----------

colLabel = "quality"
colFeatures = "features"
colPrediction = "prediction"

# COMMAND ----------

host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()

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