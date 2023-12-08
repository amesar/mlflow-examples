# Databricks notebook source
# MAGIC %md ## Start a model serving endpoint
# MAGIC
# MAGIC * [Create a new serving endpoint](https://docs.databricks.com/api/workspace/servingendpoints/create) (doc)
# MAGIC * [../includes/ModelServingClient]($../includes/ModelServingClient)
# MAGIC
# MAGIC ##### Widgets
# MAGIC
# MAGIC * `1. Registered model` - name of registerd model such as `andre_m.ml_models.diabetes_mlops`.
# MAGIC * `2. Model Serving endpoint`

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

# MAGIC %run ../includes/ModelServingClient

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
model_name = dbutils.widgets.get("1. Registered model")

dbutils.widgets.text("2. Model serving endpoint", _endpoint_name)
endpoint_name = dbutils.widgets.get("2. Model serving endpoint")

print("model_name:", model_name)
print("endpoint_name:", endpoint_name)
print("_alias:", _alias)

# COMMAND ----------

assert_widget(model_name, "1. Registered model")
assert_widget(model_name, "2. Model serving endpoint")

# COMMAND ----------

# MAGIC %md #### List endpoints

# COMMAND ----------

list_model_serving_endpoints()

# COMMAND ----------

# MAGIC %md #### See if our endpoint is running

# COMMAND ----------

endpoint = model_serving_client.get_endpoint(endpoint_name)
if endpoint:
    print(f"Endpoint '{endpoint_name}' is running")
    print(endpoint)
else:
    print(f"Endpoint '{endpoint_name}' is not running")

# COMMAND ----------

# MAGIC %md #### If our endpoint is running then exit notebook

# COMMAND ----------

if endpoint:
    dbutils.notebook.exit(0) 
print(f"About to launch endpoint '{endpoint_name}'")

# COMMAND ----------

# MAGIC %md #### Get version from model alias

# COMMAND ----------

model = mlflow_client.get_registered_model(model_name)
dump_obj(model)

# COMMAND ----------

version = model.aliases[_alias]
print("version:", version)

# COMMAND ----------

# MAGIC %md #### Define endpoint config spec

# COMMAND ----------

served_model = "my-model"
spec = {
    "name": endpoint_name,
    "config": { 
      "served_models": [ 
        { 
          "name": served_model,
          "model_name": model_name,
          f"model_version": version,
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

model_serving_client.wait_until(endpoint_name, max=60, sleep_time=10)

# COMMAND ----------

model_serving_client.get_endpoint(endpoint_name)

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC Once the serving endpoint has started, go to the **[04b_Model_Serving_Score]($04b_Model_Serving_Score)** notebook to submit scoring requests.
