# Databricks notebook source
# MAGIC %md ## Stop a model serving endpoint
# MAGIC
# MAGIC * [Delete a serving endpoint ](https://docs.databricks.com/api/workspace/servingendpoints/delete)
# MAGIC * [../includes/ModelServingClient]($../includes/ModelServingClient)

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

# MAGIC %run ../includes/ModelServingClient

# COMMAND ----------

dbutils.widgets.text("1. Model serving endpoint", _endpoint_name)
endpoint_name = dbutils.widgets.get("1. Model serving endpoint")
print("endpoint_name:", endpoint_name)

# COMMAND ----------

# MAGIC %md ### Display endpoints

# COMMAND ----------

endpoints = model_serving_client.list_endpoints()
for e in endpoints:
    print(f"{e['name']} - {e['creator']}")

# COMMAND ----------

# MAGIC %md ### Stop endpoint

# COMMAND ----------

model_serving_client.get_endpoint(endpoint_name)

# COMMAND ----------

model_serving_client.stop_endpoint(endpoint_name)

# COMMAND ----------

# MAGIC %md ### Display endpoints

# COMMAND ----------

model_serving_client.get_endpoint(endpoint_name)

# COMMAND ----------

endpoints = model_serving_client.list_endpoints()
for e in endpoints:
    print(f"{e['name']} - {e['creator']}")

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC **_Congratulations!_** You have finished your Diabetes Mini MLOps example. There is no next notebook.
