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

# MAGIC %md #### List endpoints

# COMMAND ----------

list_model_serving_endpoints()

# COMMAND ----------

# MAGIC %md #### Stop endpoint

# COMMAND ----------

model_serving_client.get_endpoint(endpoint_name)

# COMMAND ----------

model_serving_client.stop_endpoint(endpoint_name)

# COMMAND ----------

# MAGIC %md #### List endpoints

# COMMAND ----------

model_serving_client.get_endpoint(endpoint_name)

# COMMAND ----------

list_model_serving_endpoints()

# COMMAND ----------

# MAGIC %md #### No next notebook
# MAGIC
# MAGIC **_Congratulations!_** You have finished your Diabetes Mini MLOps example. There is no next notebook.
