# Databricks notebook source
def get_notebook_tag(tag):
    tag = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get(tag)
    return None if tag.isEmpty() else tag.get()

# COMMAND ----------

host_name = get_notebook_tag("browserHostName")

# COMMAND ----------

def display_run_uri(experiment, run):
    uri = f"https://{host_name}/#mlflow/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
    displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_experiment_uri(experiment):
    uri = f"https://{host_name}/#mlflow/experiments/{experiment.experiment_id}"
    displayHTML("""<b>Experiment URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

import os
import mlflow
def print_versions():
    print("Versions:")
    print("  MLflow version:", mlflow.__version__)
    print("  spark.version:", spark.version)
    print("  DATABRICKS_RUNTIME_VERSION:", os.environ.get("DATABRICKS_RUNTIME_VERSION",None))
    print()
print_versions()
