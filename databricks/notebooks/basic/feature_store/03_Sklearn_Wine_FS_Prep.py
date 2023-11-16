# Databricks notebook source
# MAGIC %md ## Sklearn feature store data preparation
# MAGIC
# MAGIC **Overview**
# MAGIC * Creates  feature table.
# MAGIC * Run this notebook once before running [Sklearn_Wine_FS]($Sklearn_Wine_FS).
# MAGIC
# MAGIC **Widgets**
# MAGIC * `1. Database` 
# MAGIC * `2. Datapath` 
# MAGIC * `3. Overwrite table`
# MAGIC * `4. Unity Catalog`

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

fs_default_datapath = "/databricks-datasets/wine-quality/winequality-white.csv"

dbutils.widgets.text("1. Database", "")
dbutils.widgets.text("2. Datapath", fs_default_datapath)
dbutils.widgets.dropdown("3. Overwrite table","yes",["yes","no"])
dbutils.widgets.dropdown("4. Unity Catalog", "no", ["yes","no"])

fs_database = dbutils.widgets.get("1. Database")
fs_datapath = dbutils.widgets.get("2. Datapath")
overwrite_table = dbutils.widgets.get("3. Overwrite table")
use_uc = dbutils.widgets.get("4. Unity Catalog") == "yes"

fs_table = f"{fs_database}.wine_features"

print("fs_database:", fs_database)
print("fs_datapath:", fs_datapath)
print("fs_table:", fs_table)
print("overwrite_table:", overwrite_table)
print("use_uc:", use_uc)

# COMMAND ----------

assert_widget(fs_database, "1. Database")

# COMMAND ----------

if use_uc:
    client = activate_unity_catalog()
    print("New client._registry_uri:",client._registry_uri)

# COMMAND ----------

# MAGIC %md ### Data prep

# COMMAND ----------

raw_df = read_wine_data(fs_datapath)
display(raw_df.limit(10))

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

def add_id_column(df, id_column_name):
    columns = df.columns
    new_df = df.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def rename_columns(df):
    new_df = df
    for column in df.columns:
        new_df = new_df.withColumnRenamed(column, column.replace(" ", "_"))
    return new_df

renamed_df = rename_columns(raw_df)
data_id_df = add_id_column(renamed_df, "wine_id")

# Drop target column ('quality') as it is not included in the feature table
features_df = data_id_df.drop("quality")

display(features_df)

# COMMAND ----------

# MAGIC %md ### Database table

# COMMAND ----------

spark.sql(f"create database if not exists {fs_database}")

# COMMAND ----------

# MAGIC %md ### Feature store client

# COMMAND ----------

from databricks.feature_store.client import FeatureStoreClient
fs_client = FeatureStoreClient()

# COMMAND ----------

# MAGIC %md ### Create feature table

# COMMAND ----------

def fs_table_exists(fs_table):
    try:
        fs_client.get_table(fs_table)
        return True
    except Exception as e: # ValueError
        print("INFO:",e,type(e))
        return False
fs_table_exists(fs_table)

# COMMAND ----------

if not fs_table_exists(fs_table):
    print(f"Creating feature table {fs_table}")
    fs_client.create_table(
        name = fs_table,
        primary_keys = ["wine_id"],
        df = features_df,
     description="id and features of all wine",
    )

# COMMAND ----------

display(spark.sql(f"describe extended {fs_table}"))
