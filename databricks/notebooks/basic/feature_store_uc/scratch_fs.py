# Databricks notebook source
# MAGIC %run ./Common

# COMMAND ----------

# MAGIC %md ### SQL

# COMMAND ----------

# MAGIC %sql use andre_fs_wine_2;
# MAGIC show tables;

# COMMAND ----------

from pyspark.sql.functions import *

spark.sql("show databases").filter(col("databaseName").like("andre\_%")).show(1000,False)


# COMMAND ----------

# MAGIC %sql drop  database andre_fs_wine cascade

# COMMAND ----------

spark.catalog.tableExists("andre_fs_wine_2.wine_features")

# COMMAND ----------

from databricks.feature_store.client import FeatureStoreClient
fs_client = FeatureStoreClient()
help(fs_client)

# COMMAND ----------

fs_client.drop_table("andre_fs_wine.wine_features")

# COMMAND ----------

# MAGIC %md ### FS

# COMMAND ----------

table_name = "andre_fs_wine.wine_features"
lookup_key = "wine_id"

# COMMAND ----------

from databricks.feature_store import FeatureLookup

model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
len(model_feature_lookups)

# COMMAND ----------

lk = model_feature_lookups[0]
lk

# COMMAND ----------

dump_obj(lk)

# COMMAND ----------


