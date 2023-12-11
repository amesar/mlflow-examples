# Databricks notebook source
default_model_name = "databricks_llama_2_models.models.llama_2_7b_chat_hf" # per Marketplace notebook
#default_model_name = "marketplace_staging_llama_2_models.models.llama_2_7b_chat_hf" # e2_dogfood

print("default_model_name:", default_model_name)

# COMMAND ----------

def assert_widget(value, name):
    if len(value.rstrip())==0:
        raise Exception(f"ERROR: '{name}' widget is required")

# COMMAND ----------

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host_name = ctx.tags().get("browserHostName").get()
token = ctx.apiToken().get()

# COMMAND ----------

import os

def mk_absolute_path(path):
    """
    https://docs.databricks.com/en/files/workspace-interact.html
    Returns: 
      repo:      /Workspace/Repos/Users/andre@databricks.com/mlflow-examples/notebooks/llama2/questions.csv'
      non-repo: '/Workspace/Users/andre@databricks.com/work/llama2/questions.csv'
    """
    if os.path.isabs(path):
        return path
    path = os.path.join(os.getcwd(), path)
    return f"file:{path}"

# COMMAND ----------

from pyspark.sql.types import *

def load_from_path(path):
    print(f"Reading from file '{path}'")
    path = mk_absolute_path(path)
    print(f"Reading from file '{path}'")
    schema = StructType([StructField("question", StringType(), True)])
    return (spark.read.format("csv")
      .option("header", False)
      .schema(schema)
      .load(path))

# COMMAND ----------

def load_data(name):
    toks = name.split(".")
    if len(toks) == 3: # If unity catalog 3 component name
        print(f"Reading from table '{name}'")
        return spark.table(name)
    else: # otherwise assume its a CSV file
        return load_from_path(name)

# COMMAND ----------

def dump(dct, title=None, sort_keys=None, indent=2):
    if title:
        print(f"{title}:")
    print(json.dumps(dct, sort_keys=sort_keys, indent=indent))

# COMMAND ----------

import mlflow
import os
print("MLflow versions:", mlflow.__version__)
print("DBR version     ", os.environ.get("DATABRICKS_RUNTIME_VERSION"))
