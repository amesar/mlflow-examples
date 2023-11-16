# Databricks notebook source
# MAGIC %run ./http_client

# COMMAND ----------

# MAGIC %md #### HttpClient

# COMMAND ----------

http_client = HttpClient("")
http_client

# COMMAND ----------

clusters = http_client.get("2.0/clusters/list")
clusters = clusters["clusters"]
len(clusters)

# COMMAND ----------

# MAGIC %md #### DatabricksHttpClient

# COMMAND ----------

dbx_client = DatabricksHttpClient()
dbx_client

# COMMAND ----------

clusters = dbx_client.get("clusters/list")
clusters = clusters["clusters"]
len(clusters)

# COMMAND ----------

clusters[0]

# COMMAND ----------

# MAGIC %md #### MlflowHttpClient

# COMMAND ----------

mlflow_client = MlflowHttpClient()
mlflow_client

# COMMAND ----------

models = mlflow_client.get("registered-models/search", {"max-results": 2000})
models = models["registered_models"]
len(models)

# COMMAND ----------

for m in models[:10]:
    print(m["name"])

# COMMAND ----------

models[0]

# COMMAND ----------


