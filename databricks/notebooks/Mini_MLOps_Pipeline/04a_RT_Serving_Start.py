# Databricks notebook source
# MAGIC %md ## Start a model serving endpoint

# COMMAND ----------

# MAGIC %md ### Setup includes

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

# MAGIC %md ### Setup widgets

# COMMAND ----------

"""
dbutils.widgets.text("1. Endpoint", _endpoint_name)
endpoint_name = dbutils.widgets.get("1. Endpoint")

dbutils.widgets.text("2. Served Model", "my_model")
served_model = dbutils.widgets.get("2. Served Model")

dbutils.widgets.text("3. Registered Model", _model_name)
registered_model = dbutils.widgets.get("3. Registered Model")

dbutils.widgets.text("4. Registered Model Version", "1")
registered_model_version = dbutils.widgets.get("4. Registered Model Version")

print("endpoint_name:",endpoint_name)
print("served_model:",served_model)
print("registered_model:",registered_model)
print("registered_model_version:",registered_model_version)

if endpoint_name =="": raise Exception("Missing 'Endpoint'")
if served_model =="": raise Exception("Missing 'Served Model'")
if registered_model=="": raise Exception("Missing 'Registered Model'")
"""

# COMMAND ----------

dbutils.widgets.text("Registered Model Version", "1")
registered_model_version = dbutils.widgets.get("Registered Model Version")
print("registered_model_version:", registered_model_version)

# COMMAND ----------

# MAGIC %md #### List all endpoints

# COMMAND ----------

endpoints = model_serving_client.list_endpoints()
for e in endpoints:
    print(f"  {e['name']} - {e['creator']}")

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
    "name": f"{_endpoint_name}",
    "config": { 
      "served_models": [ 
        { 
          "name": f"{served_model}",
          "model_name": f"{_model_name}",
          f"model_version": f"{registered_model_version}",
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

model_serving_client.wait_until(_endpoint_name, max=50, sleep_time=4)

# COMMAND ----------

model_serving_client.get_endpoint(_endpoint_name)

# COMMAND ----------


