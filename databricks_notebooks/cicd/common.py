# Databricks notebook source
import os
import requests

def download_uri(uri, path):
    if os.path.exists(path):
        print(f"File {path} already exists")
    else:
        print(f"Downloading {uri} to {path}")
        rsp = requests.get(uri)
        with open(path, "w") as f:
            f.write(rsp.text)

# COMMAND ----------

def download_wine_file():
    data_path = "/dbfs/tmp/mlflow_wine-quality.csv"
    uri = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
    download_uri(uri, data_path)
    return data_path