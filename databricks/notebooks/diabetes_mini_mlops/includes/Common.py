# Databricks notebook source
# Common helper functions. 
# Global variables:
#  * model_uri - URI of the registered model.
#  * experiment_name - name of the 01_Train_Model notebook experiment. 
#  * endpoint_name - mini_mlops

# COMMAND ----------

# MAGIC %run ./Load_Data

# COMMAND ----------

import os
import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow_client = mlflow.MlflowClient()

# COMMAND ----------

print("Versions:")
print("  MLflow Version:", mlflow.__version__)
print("  DATABRICKS_RUNTIME_VERSION:", os.environ.get('DATABRICKS_RUNTIME_VERSION'))
print("  mlflow.get_registry_uri:", mlflow.get_registry_uri())

# COMMAND ----------

# Experiment name is the same as the 01_Train_Model notebook.
_notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

_notebook = _notebook_context.notebookPath().get()
_dir = os.path.dirname(_notebook)
_experiment_name = os.path.join(_dir, "01_Train_Model")
_alias = "champ"
_endpoint_name = "diabetes_mini_mlops"

print("_experiment_name:", _experiment_name)
print("_alias:", _alias)

# COMMAND ----------

def _get_notebook_tag(tag):
    tag = _notebook_context.tags().get(tag)
    return None if tag.isEmpty() else tag.get()

# COMMAND ----------

def _create_experiment_for_repos():
    print("Creating Repos scratch experiment")
    with mlflow.start_run() as run:
        pass # mlflow.set_tag("info","hi there")
    return mlflow_client.get_experiment(run.info.experiment_id)

# COMMAND ----------

# Gets the current notebook's experiment information

def get_experiment():
    experiment_name = _notebook
    print("Experiment name:", experiment_name)
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    # Is running in as Repos, cannot need to create its default "notebook" experiment
    if not experiment:
        experiment = _create_experiment_for_repos()
        print(f"Running as Repos - created Repos experiment:", experiment.name)

    print("Experiment ID:", experiment.experiment_id)
    print("Experiment name:", experiment.name)
    return experiment

# COMMAND ----------

def delete_runs(experiment):
    runs = mlflow_client.search_runs(experiment.experiment_id)
    print(f"Found {len(runs)} runs for experiment_id {experiment.experiment_id} - {experiment.name}")
    for run in runs:
        mlflow_client.delete_run(run.info.run_id)

# COMMAND ----------

def assert_widget(value, name):
    if len(value.rstrip())==0:
        raise Exception(f"ERROR: '{name}' widget is required")

# COMMAND ----------

# Create registered model 
# If the model already exists remove, then delete its existing versions.

def create_registered_model(model_name, delete_registered_model=True):
    from mlflow.exceptions import MlflowException, RestException

    try:
        registered_model = mlflow_client.get_registered_model(model_name)
        print(f"Found registered model '{model_name}'")
        if delete_registered_model:
            versions = mlflow.search_model_versions(filter_string=f"name='{model_name}'")
            print(f"Found {len(versions)} model versions to delete")
            for vr in versions:
                print(f"  Deleting version={vr.version} status={vr.status} stage={vr.current_stage} run_id={vr.run_id}")
                mlflow_client.delete_model_version(model_name, vr.version)
            print(f"Deleting registered model '{model_name}'")
            mlflow_client.delete_registered_model(model_name)
            registered_model = mlflow_client.create_registered_model(model_name)
    except RestException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            print(f"Creating registered model '{model_name}'")
            registered_model = mlflow_client.create_registered_model(model_name)
        else:
            raise e

# COMMAND ----------

_host_name = _get_notebook_tag("browserHostName")
print("_host_name:", _host_name)

# COMMAND ----------

_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg",f"[DEFAULT]\nhost=https://{_host_name}\ntoken = "+_token,overwrite=True)

# COMMAND ----------

def display_experiment_uri(experiment):
    host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
    uri = f"https://{host_name}/#mlflow/experiments/{experiment.experiment_id}"
    displayHTML(f"""
    <table cellpadding=5 cellspacing=0 border=1 bgcolor="#FDFEFE" style="font-size:13px;">
    <tr><td colspan=2><b><i>Experiment</i></b></td></tr>
    <tr><td>UI link</td><td><a href="{uri}">{uri}</a></td></tr>
    <tr><td>Name</td><td>{experiment.name}</td></tr>
    <tr><td>ID</td><td>{experiment.experiment_id}</td></tr>
    </table>
    """)

# COMMAND ----------

def display_registered_model_uri(model_name):
    uri = f"https://{_host_name}/#mlflow/models/{model_name}"
    displayHTML("""<b>Registered Model URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

"""
Waits until a model version is in the READY status.
"""
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_version_ready(model_name, model_version, sleep_time=1, iterations=100):
    start = time.time()
    for _ in range(iterations):
        version = mlflow_client.get_model_version(model_name, model_version.version)
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

def dump_obj(obj, title=None):
    title = title if title else type(obj).__name__
    print(title)
    for k,v in obj.__dict__.items():
        print(f"  {k}: {v}")
