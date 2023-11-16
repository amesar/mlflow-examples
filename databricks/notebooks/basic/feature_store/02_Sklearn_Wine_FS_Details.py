# Databricks notebook source
# MAGIC %md ## Sklearn Wine Feature Store - Details
# MAGIC
# MAGIC Get details of feature table.
# MAGIC
# MAGIC Widgets:
# MAGIC * `1. Feature table`

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Feature table", "")

fs_table_name = dbutils.widgets.get("1. Feature table")

print("fs_table_name:", fs_table_name)

# COMMAND ----------

assert_widget(fs_table_name, "1. Feature table")

# COMMAND ----------

#toggle_unity_catalog(fs_table_name)

# COMMAND ----------

mlflow.get_tracking_uri()

# COMMAND ----------

# MAGIC %md #### Get FeatureStoreClient

# COMMAND ----------

from databricks.feature_store.client import FeatureStoreClient
fs_client = FeatureStoreClient()
dump_obj(fs_client)

# COMMAND ----------

fs_table = fs_client.get_table(fs_table_name)
fs_table

# COMMAND ----------

# MAGIC %md #### Get table

# COMMAND ----------

dump_obj(fs_table)

# COMMAND ----------

# MAGIC %md #### Describe feature table

# COMMAND ----------

display(spark.sql(f"describe extended {fs_table_name}"))
