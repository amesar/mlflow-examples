# Databricks notebook source
# MAGIC %md ## Real-time Model Serving Llama 2
# MAGIC
# MAGIC #### Overview
# MAGIC * Launches a model serving endpoint with the REST API.
# MAGIC * Sends questions to be scored to the endpoint.
# MAGIC * WIP: Creating correct request to model serving endpoint in the works.
# MAGIC * Assuming e2-dogfood model: `marketplace_staging_llama_2_models.models.llama_2_7b_chat_hf`
# MAGIC
# MAGIC #### Docs
# MAGIC * https://docs.databricks.com/api/workspace/servingendpoints
# MAGIC * https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu
# MAGIC * [Send scoring requests to serving endpoints](https://docs.databricks.com/en/machine-learning/model-serving/score-model-serving-endpoints.html)
# MAGIC
# MAGIC #### Widget values
# MAGIC ##### _Workload type_
# MAGIC
# MAGIC [GPU types](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu):
# MAGIC * GPU
# MAGIC * GPU_MEDIUM - works fine
# MAGIC * GPU_MEDIUM_4
# MAGIC * GPU_MEDIUM_8
# MAGIC * GPU_LARGE_8
# MAGIC
# MAGIC ##### _Workload size_
# MAGIC * Small - works fine
# MAGIC * Medium 
# MAGIC * Large
# MAGIC
# MAGIC ##### Last updated: _2023-12-10_

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

# MAGIC %run ../includes/ModelServingClient

# COMMAND ----------

dbutils.widgets.text("1. Model", default_model_name)
dbutils.widgets.text("2. Version", "1")
dbutils.widgets.text("3. Endpoint", "llama2_simple")
dbutils.widgets.text("4. Workload type", "GPU_MEDIUM")
dbutils.widgets.text("5. Workload size", "Small")
dbutils.widgets.text("6. Max tokens", "128")

model_name = dbutils.widgets.get("1. Model")
version = dbutils.widgets.get("2. Version")
endpoint_name = dbutils.widgets.get("3. Endpoint")
workload_type = dbutils.widgets.get("4. Workload type")
workload_size = dbutils.widgets.get("5. Workload size")
max_tokens = dbutils.widgets.get("6. Max tokens")

print("model:", model_name)
print("version:", version)
print("endpoint_name:", endpoint_name)
print("workload_type:", workload_type)
print("workload_size:", workload_size)
print("max_tokens:", max_tokens)

# COMMAND ----------

assert_widget(model_name, "1. Model name")
assert_widget(version, "2. Version")
assert_widget(workload_type, "4. Workload type")
assert_widget(workload_size, "5. Workload size")

# COMMAND ----------

model_uri = f"models:/{model_name}/{version}"
model_uri

# COMMAND ----------

# MAGIC %md #### Define endpoint spec

# COMMAND ----------

served_model = "my-model"
spec = {
    "name": endpoint_name,
    "config": { 
      "served_models": [ 
        { 
          "name": "mi-llamita",
          "model_name": model_name,
          "model_version": version,
          "workload_size": "Medium",
          "scale_to_zero_enabled": False,
          "workload_type": workload_type,
        } 
      ] 
    } 
}
spec

# COMMAND ----------

# MAGIC %md #### Start the endpoint
# MAGIC
# MAGIC

# COMMAND ----------

model_serving_client.start_endpoint(spec)

# COMMAND ----------

# MAGIC %md #### Wait until endpoint is in READY state
# MAGIC * This can take up to 10 minutes.

# COMMAND ----------

model_serving_client.wait_until(endpoint_name, max=120, sleep_time=10)

# COMMAND ----------

# MAGIC %md #### Get endpoint info

# COMMAND ----------

model_serving_client.get_endpoint(endpoint_name)

# COMMAND ----------

# MAGIC %md #### Create the questions
# MAGIC * Several different input formats are supported: 
# MAGIC   * input
# MAGIC   * instances
# MAGIC   * dataframe_records
# MAGIC   * dataframe_split
# MAGIC
# MAGIC See documentaion [Send scoring requests to serving endpoints](https://docs.databricks.com/en/machine-learning/model-serving/score-model-serving-endpoints.html).

# COMMAND ----------

import json

# COMMAND ----------

def as_dataframe_records(questions):
    return {
        "dataframe_records": [ { "prompt": q } for q in questions],
        "params": {
                "temperature": 0.5,
        "max_tokens": max_tokens
        }
    }

# COMMAND ----------

def as_inputs(questions):
    return {
        "inputs": {
            "prompt": questions,
        },
        "params": {
            "temperature": 0.5,
            "max_tokens": max_tokens
        }
    }

# COMMAND ----------

questions = [
  "What is the southern most town in the world? How do I get there?",
  "What is northern most town in the world?",
  "What is the eastern most town in the world?",
  "What is the western most town in the world?"
]

request = as_inputs(questions)
#request = as_dataframe_records(questions)
dump(request)

# COMMAND ----------

# MAGIC %md #### Call Model serving endpoint

# COMMAND ----------

endpoint_uri = f"https://{host_name}/serving-endpoints/{endpoint_name}/invocations"
endpoint_uri

# COMMAND ----------

import requests

headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
response = requests.post(endpoint_uri, headers=headers, json=request, timeout=15)
dump(response.json())
