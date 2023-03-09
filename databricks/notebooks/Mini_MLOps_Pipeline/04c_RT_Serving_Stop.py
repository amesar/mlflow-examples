# Databricks notebook source
# MAGIC %md ## Stop a model serving endpoint

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

model_serving_client.get_endpoint(_endpoint_name)

# COMMAND ----------

model_serving_client.stop_endpoint(_endpoint_name)

# COMMAND ----------

model_serving_client.get_endpoint(_endpoint_name)

# COMMAND ----------

endpoints = model_serving_client.list_endpoints()
for e in endpoints:
    print(f"  {e['name']} - {e['creator']}")