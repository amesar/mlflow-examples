# Databricks notebook source
# MAGIC %md ## Score a model with RT Model Serving

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

# Example: https://e2-catfood.cloud.mycompany.com/serving-endpoints/mini_mlops/invocations

endpoint_uri = f"https://{_host_name}/serving-endpoints/{_endpoint_name}/invocations"
import os
os.environ["ENDPOINT_URI"] = endpoint_uri
os.environ["TOKEN"] = token
endpoint_uri

# COMMAND ----------

# MAGIC %md ### Score with curL

# COMMAND ----------

# MAGIC %sh echo $ENDPOINT_URI

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -s $ENDPOINT_URI \
# MAGIC -H "Authorization: Bearer $TOKEN" \
# MAGIC -H 'Content-Type: application/json' \
# MAGIC -d '{"dataframe_split": { "columns": [ "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol" ], "data": [ [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8 ], [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5 ], [ 8.1, 0.28, 0.4,   6.9, 0.05,  30,  97, 0.9951, 3.26, 0.44, 10.1 ] ] } }
# MAGIC '

# COMMAND ----------

# MAGIC %md ### Score with Python requests

# COMMAND ----------

import requests
data = { "dataframe_split": { 
    "columns": 
        [ "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol" ], 
     "data": [ 
         [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8 ], 
         [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5 ], 
         [ 8.1, 0.28, 0.4,   6.9,  0.05, 30,  97, 0.9951, 3.26, 0.44, 10.1 ] 
      ] 
    } 
}
import json
data = json.dumps(data)

headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
rsp = requests.post(endpoint_uri, headers=headers, data=data, timeout=15)
rsp.text

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC When finished scoring, go to the **[04c_RT_Serving_Stop]($04c_RT_Serving_Stop)** notebook to shut down the serving endpoint.
