# Databricks notebook source
# MAGIC %md ## Score a model with model serving endpoint
# MAGIC
# MAGIC * [Send scoring requests to serving endpoints](https://docs.databricks.com/en/machine-learning/model-serving/score-model-serving-endpoints.html)

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
data = { "dataframe_split": { 
    "columns": 
        [ "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6" ], 
     "data": [ 
         [0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.02, -0.018],  
         [-0.002, -0.045, -0.051, -0.026, -0.008, -0.019, 0.074, -0.039, -0.068, -0.092]
      ] 
    } 
}
import json
data = json.dumps(data)

headers = { "Authorization": f"Bearer {_token}", "Content-Type": "application/json" }
rsp = requests.post(endpoint_uri, headers=headers, data=data, timeout=15)
rsp.text

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC When finished scoring, go to the **[04c_Model_Serving_Stop]($04c_Model_Serving_Stop)** notebook to shut down the serving endpoint.
