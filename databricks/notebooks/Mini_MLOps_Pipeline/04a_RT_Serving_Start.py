# Databricks notebook source
# MAGIC %md ## Start a model serving endpoint
# MAGIC
# MAGIC * https://docs.databricks.com/api/workspace/servingendpoints/create
# MAGIC * [../includes/ModelServingClient]($../includes/ModelServingClient)
# MAGIC

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

# MAGIC %run ../includes/ModelServingClient

# COMMAND ----------

dbutils.widgets.text("Registered Model Version", "1")
registered_model_version = dbutils.widgets.get("Registered Model Version")
print("registered_model_version:", registered_model_version)

# COMMAND ----------

# MAGIC %md #### List all endpoints

# COMMAND ----------

import pandas as pd

endpoints = model_serving_client.list_endpoints()
if len(endpoints) > 0:
    lst = [ ( e["name"], e["creator"] ) for e in endpoints ]
    df = pd.DataFrame(lst, columns = ["Name","Creator"])
    display(spark.createDataFrame(df))

# COMMAND ----------

# MAGIC %md #### See if our endpoint is running

# COMMAND ----------

endpoint = model_serving_client.get_endpoint(_endpoint_name)
if endpoint:
    print(f"Endpoint '{_endpoint_name}' is running")
    print(endpoint)
else:
    print(f"Endpoint '{_endpoint_name}' is not running")

# COMMAND ----------

# MAGIC %md #### If our endpoint is running then exit notebook

# COMMAND ----------

if endpoint:
    dbutils.notebook.exit(0) 
print(f"About to launch endpoint '{_endpoint_name}'")

# COMMAND ----------

# MAGIC %md #### Define endpoint spec

# COMMAND ----------

served_model = "my-model"
spec = {
    "name": _endpoint_name,
    "config": { 
      "served_models": [ 
        { 
          "name": served_model,
          "model_name": _model_name,
          f"model_version": registered_model_version,
          "workload_size": "Small",
          "scale_to_zero_enabled": False
        } 
      ] 
    } 
}
spec

# COMMAND ----------

# MAGIC %md #### Start the endpoint

# COMMAND ----------

model_serving_client.start_endpoint(spec)

# COMMAND ----------

# MAGIC %md #### Wait until endpoint is in READY state

# COMMAND ----------

model_serving_client.wait_until(_endpoint_name, max=60, sleep_time=10)

# COMMAND ----------

model_serving_client.get_endpoint(_endpoint_name)

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC Once the serving endpoint has started, go to the **[04b_RT_Serving_Score]($04b_RT_Serving_Score)** notebook to submit scoring requests.
