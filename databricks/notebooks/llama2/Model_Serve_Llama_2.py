# Databricks notebook source
# MAGIC %md ## Real-time Model Serving Llama 2
# MAGIC
# MAGIC * https://databricks-sdk-py.readthedocs.io/en/latest/workspace/serving_endpoints.html
# MAGIC * https://docs.databricks.com/api/workspace/servingendpoints

# COMMAND ----------

import os
from databricks import sdk
print("DBR:        ", os.environ.get("DATABRICKS_RUNTIME_VERSION",None))
print("sdk.version:", sdk.version.__version__)

# COMMAND ----------

# MAGIC %md #### Install latest SDK
# MAGIC
# MAGIC * DBR ML 14.1 has SDK 0.1.6 (2023-05-10) installed as default which does not support auto-authentication.

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

default_model_name = "marketplace_staging_llama_2_models.models.llama_2_7b_chat_hf"

dbutils.widgets.text("1. Model", default_model_name)
dbutils.widgets.text("2. Version", "1")
dbutils.widgets.text("3. Endpoint", "llama2_simple")

model_name = dbutils.widgets.get("1. Model")
version = dbutils.widgets.get("2. Version")
endpoint_name = dbutils.widgets.get("3. Endpoint")

print("model:", model_name)
print("version:", version)
print("endpoint_name:", endpoint_name)

# COMMAND ----------

assert_widget(model_name, "1. Model name")
assert_widget(version, "2. Version")
assert_widget(endpoint_name, "3. Endpoint")

# COMMAND ----------

model_uri = f"models:/{model_name}/{version}"
model_uri

# COMMAND ----------

from databricks.sdk import WorkspaceClient
client = WorkspaceClient()

# COMMAND ----------

# MAGIC %md #### Define endpoint spec

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

spec = ServedModelInput(
    name="my-llama",
    model_name = model_name,
    model_version = version,
    workload_size = "Small",
    workload_type = "GPU",
    scale_to_zero_enabled = True
)
spec

# COMMAND ----------

# MAGIC %md #### Start the endpoint
# MAGIC
# MAGIC * Note that `serving_endpoints.create_and_wait` is not documented 
# MAGIC * https://databricks-sdk-py.readthedocs.io/en/latest/workspace/serving_endpoints.html

# COMMAND ----------

#client.serving_endpoints.create_and_wait(

client.serving_endpoints.create(
    name = endpoint_name,
    config = EndpointCoreConfigInput(served_models=[spec])
)

# COMMAND ----------

# MAGIC %md #### List endpoints

# COMMAND ----------

endpoints = client.serving_endpoints.list()
for e in endpoints:
    print(e.name)

# COMMAND ----------

# MAGIC %md #### Get endpoint info

# COMMAND ----------

endpoint = client.serving_endpoints.get(endpoint_name)
endpoint.as_dict()

# COMMAND ----------

# MAGIC %md #### Wait until the serving endpoint is ready

# COMMAND ----------

import time
from databricks.sdk.service.serving import EndpointStateReady

def wait_until_ready(max_calls, seconds=20):
    for j in range(0,max):
        client.serving_endpoints.get(endpoint_name)
        print(f"{j+1}/{max_calls}: state.ready: {endpoint.state.ready}  state.config_update={endpoint.state.config_update}")
        if endpoint.state.ready != EndpointStateReady.NOT_READY:
            print("READY")
            break
        time.sleep(seconds)
        
wait_until_ready(50, 20)

# COMMAND ----------

# MAGIC %md #### Make questions

# COMMAND ----------

import pandas as pd
import json

def mk_questions(questions):
    questions = [ [q] for q in questions ]
    pdf = pd.DataFrame(questions, columns = ["question"])
    ds_dict = {"dataframe_split": pdf.to_dict(orient="split")}
    return json.dumps(ds_dict, allow_nan=True)

# COMMAND ----------

questions = [
  "What is the southern most town in the world? How do I get there?",
  "What is northern most town in the world?",
  "What is the eastern most town in the world?",
  "What is the western most town in the world?"
]

questions = mk_questions(questions)
questions

# COMMAND ----------

# MAGIC %md #### Call Model Server

# COMMAND ----------

answers = client.serving_endpoints.query(
    name = endpoint_name,
    dataframe_records=questions
)
answers
