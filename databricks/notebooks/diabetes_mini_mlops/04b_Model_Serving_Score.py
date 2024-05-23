# Databricks notebook source
# MAGIC %md ## Score a model with serving endpoint
# MAGIC
# MAGIC ##### Overview
# MAGIC * Example of using the following request formats:
# MAGIC     * JSON `dataframe_split` 
# MAGIC     * JSON dataframe_records` format (not recommended)
# MAGIC     * CSV
# MAGIC
# MAGIC ##### Databricks Documentations:
# MAGIC * [Querying methods and examples](https://docs.databricks.com/en/machine-learning/model-serving/score-custom-model-endpoints.html#querying-methods-and-examples) - Query serving endpoints for custom models
# MAGIC
# MAGIC
# MAGIC ##### MLflow Documentations:
# MAGIC * [Accepted Input Formats](https://mlflow.org/docs/latest/deployment/deploy-model-locally.html?highlight=json%20scoring%20formats#accepted-input-formats) - Deploy MLflow Model as a Local Inference Server
# MAGIC
# MAGIC ##### Widgets
# MAGIC * `1. Model Serving endpoint`

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

dbutils.widgets.text("1. Model serving endpoint", _endpoint_name)
endpoint_name = dbutils.widgets.get("1. Model serving endpoint")
print("endpoint_name:", endpoint_name)

# COMMAND ----------

# Example: https://e2-catfood.cloud.mycompany.com/serving-endpoints/mini_mlops/invocations

endpoint_uri = f"https://{_host_name}/serving-endpoints/{endpoint_name}/invocations"
import os
os.environ["ENDPOINT_URI"] = endpoint_uri
os.environ["TOKEN"] = _token
endpoint_uri

# COMMAND ----------

# MAGIC %md ### Score with curl

# COMMAND ----------

# MAGIC %sh echo $ENDPOINT_URI

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -s $ENDPOINT_URI \
# MAGIC -H "Authorization: Bearer $TOKEN" \
# MAGIC -H 'Content-Type: application/json' \
# MAGIC -d '{ "dataframe_split": { "columns": [ "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6" ], "data": [ [ 0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.02, -0.018 ] ] } }
# MAGIC '

# COMMAND ----------

# MAGIC %md ### Score with Python requests

# COMMAND ----------

import requests
import json
headers = { "Authorization": f"Bearer {_token}", "Content-Type": "application/json" }

# COMMAND ----------

# MAGIC %md ##### Pandas `dataframe_split` format

# COMMAND ----------

data = { "dataframe_split": { 
    "columns": 
        [ "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6" ], 
     "data": [ 
         [0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.02, -0.018],  
         [-0.002, -0.045, -0.051, -0.026, -0.008, -0.019, 0.074, -0.039, -0.068, -0.092]
      ] 
    } 
}
rsp = requests.post(endpoint_uri, headers=headers, json=data, timeout=15)
rsp.text

# COMMAND ----------

# MAGIC %md ##### Pandas `dataframe_records` format

# COMMAND ----------

data = {
  "dataframe_records": [
    {
      "age": 0.038,
      "sex": 0.051,
      "bmi": 0.062,
      "bp": 0.022,
      "s1": -0.044,
      "s2": -0.035,
      "s3": -0.043,
      "s4": -0.003,
      "s5": 0.02,
      "s6": -0.018
    },
    {
      "age": -0.002,
      "sex": -0.045,
      "bmi": -0.051,
      "bp": -0.026,
      "s1": -0.008,
      "s2": -0.019,
      "s3": 0.074,
      "s4": -0.039,
      "s5": -0.068,
      "s6": -0.092
    }
  ]
}
rsp = requests.post(endpoint_uri, headers=headers, json=data, timeout=15)
rsp.text

# COMMAND ----------

# MAGIC %md ##### CSV format

# COMMAND ----------

data = ''' "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6" 
0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.02, -0.018]
'''
headers = { "Authorization": f"Bearer {_token}", "Content-Type": "text/csv" }
rsp = requests.post(endpoint_uri, headers=headers, data=data, timeout=15)
rsp.text

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC When finished scoring, go to the **[04c_Model_Serving_Stop]($04c_Model_Serving_Stop)** notebook to shut down the serving endpoint.
