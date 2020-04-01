# Databricks notebook source
import os
import requests

def download_file(data_uri, data_path):
    if os.path.exists(data_path):
        print("File {} already exists".format(data_path))
    else:
        print("Downloading {} to {}".format(data_uri,data_path))
        rsp = requests.get(data_uri)
        with open(data_path, 'w') as f:
            f.write(requests.get(data_uri).text)

# COMMAND ----------

def download_wine_file():
    data_path = "/dbfs/tmp/mlflow_wine-quality.csv"
    data_uri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
    download_file(data_uri, data_path)
    return data_path

# COMMAND ----------

#def get_user():
  #user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
  #return user
  
def get_experiment_name():
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()