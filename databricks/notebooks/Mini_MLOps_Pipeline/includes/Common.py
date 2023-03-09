# Databricks notebook source
# Common helper functions. 
# Global variables:
#  * model_uri - URI of the registered model.
#  * experiment_name - name of the 01_Train_Model notebook experiment. 
#  * endpoint_name - mini_mlops_wine_quality

# COMMAND ----------

_model_name = "mini_mlops_pipeline"
_model_uri = f"models:/{_model_name}/production"
_endpoint_name = "mini_mlops_wine_quality"
print("_model_name:", _model_name)
print("_endpoint_name:", _endpoint_name)
print("_model_uri:", _model_uri)

# COMMAND ----------

# Experiment name is the same as the 01_Train_Model notebook.

import os
notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
dir = os.path.dirname(notebook)
_experiment_name = f"{dir}/01_Train_Model"
print("_experiment_name:", _experiment_name)

# COMMAND ----------

def _get_notebook_tag(tag):
    tag = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get(tag)
    return None if tag.isEmpty() else tag.get()

# COMMAND ----------

# Gets the current notebook's experiment information

def init():
    experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    print("Experiment name:",experiment_name)
    experiment = client.get_experiment_by_name(experiment_name)
    print("Experiment ID:",experiment.experiment_id)
    return experiment.experiment_id, experiment_name

# COMMAND ----------

def delete_runs(experiment_id):
    runs = client.search_runs(experiment_id)
    print(f"Found {len(runs)} runs for experiment_id {experiment_id}")
    for run in runs:
        client.delete_run(run.info.run_id)

# COMMAND ----------

def _read_data(data_path):
    import pandas as pd
    print(f"Reading data from '{data_path}'")
    pdf = pd.read_csv(data_path)
    pdf.columns = pdf.columns.str.replace(" ","_") # make Spark legal column names
    return pdf
  
def get_wine_quality_data(table_name=""):
    """ Read CSV data from internet or from Delta table """
    data_path = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
    if table_name == "":
        return _read_data(data_path)
    else:
        if not spark.catalog._jcatalog.tableExists(table_name): # Create table if it does not exist
            print(f"Creating table '{table_name}'")
            pdf = _read_data(data_path)
            df = spark.createDataFrame(pdf)
            df.write.mode("overwrite").saveAsTable(table_name)
        print(f"Reading from table '{table_name}'")
        return spark.table(table_name).toPandas() # Read from Delta table

# COMMAND ----------

# Columns 
_col_label = "quality"
_col_prediction = "prediction"
print("_col_label:", _col_label)
print("_col_prediction:", _col_prediction)

# COMMAND ----------

# Version information
import os
import mlflow
import mlflow.spark
import pyspark
print("MLflow Version:", mlflow.__version__)
print("Spark Version:", spark.version)
print("PySpark Version:", pyspark.__version__)
print("sparkVersion:", _get_notebook_tag("sparkVersion"))
print("DATABRICKS_RUNTIME_VERSION:", os.environ.get('DATABRICKS_RUNTIME_VERSION',None))

client = mlflow.client.MlflowClient()

# COMMAND ----------

_host_name = _get_notebook_tag("browserHostName")
print("host_name:", _host_name)

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg",f"[DEFAULT]\nhost=https://{_host_name}\ntoken = "+token,overwrite=True)

# COMMAND ----------

def display_run_uri(experiment_id, run_id):
    uri = f"https://{_host_name}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
    displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_experiment_uri(experiment_id):
    uri = "https://{}/#mlflow/experiments/{}".format(_host_name, experiment_id)
    displayHTML("""<b>Experiment URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_registered_model_uri(model_name):
    uri = f"https://{_host_name}/#mlflow/models/{model_name}"
    displayHTML("""<b>Registered Model URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

"""
Wait function due to eventual consistency. 
Waits until a version is in the READY status.
"""
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
        elif status == ModelVersionStatus.FAILED_REGISTRATION:
            raise Exception(f"ERROR: status={ModelVersionStatus.to_string(status)}")
        time.sleep(sleep_time)
    end = time.time()
    print(f"Waited {round(end-start,2)} seconds")

# COMMAND ----------

# MAGIC %run ./HttpClient

# COMMAND ----------

databricks_client = DatabricksHttpClient()

# COMMAND ----------

# MAGIC %run ./ModelServingClient