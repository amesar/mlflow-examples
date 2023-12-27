# Databricks notebook source
# MAGIC %md ## Sklearn feature store data preparation
# MAGIC
# MAGIC ##### Overview
# MAGIC * Creates  feature table.
# MAGIC * Run this notebook once before running [03_Sklearn_Wine_FS]($03_Sklearn_Wine_FS).
# MAGIC * Creates table `andre_fs_wine` in specified database.
# MAGIC
# MAGIC ##### Widgets
# MAGIC * `1. Database` 
# MAGIC * `2. Datapath` - /databricks-datasets/wine-quality/winequality-white.csv
# MAGIC * `3. Overwrite feature table`
# MAGIC * `4. Drop feature table`

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

fs_default_datapath = "/databricks-datasets/wine-quality/winequality-white.csv"

dbutils.widgets.text("1. Database", "")
dbutils.widgets.text("2. Datapath", fs_default_datapath)
dbutils.widgets.dropdown("3. Overwrite table", "yes", ["yes","no"])
dbutils.widgets.dropdown("4. Drop table", "no", ["yes","no"])

fs_database = dbutils.widgets.get("1. Database")
fs_datapath = dbutils.widgets.get("2. Datapath")
overwrite_table = dbutils.widgets.get("3. Overwrite table") == "yes"
drop_table = dbutils.widgets.get("4. Drop table") == "yes"

fs_table = f"{fs_database}.wine_features"

print("fs_database:", fs_database)
print("fs_datapath:", fs_datapath)
print("fs_table:", fs_table)
print("overwrite_table:", overwrite_table)
print("drop_table:", drop_table)

# COMMAND ----------

assert_widget(fs_database, "1. Database")

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

from databricks.feature_engineering.client import FeatureEngineeringClient
fe_client = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md ### Create feature table

# COMMAND ----------

if drop_table:
    drop_fs_table(fs_table)

# COMMAND ----------

if not fs_table_exists(fs_table):
    print(f"Creating feature table '{fs_table}'")
    fe_client.create_table(
        name = fs_table,
        primary_keys = ["wine_id"],
        df = features_df,
     description="id and features of all wine",
    )

# AnalysisException: Constraint 'wine_features_pk' already exists in database 'andre_catalog'.'fs_wine'

# COMMAND ----------

# MAGIC %md ### Describe feature table

# COMMAND ----------

display(spark.sql(f"describe extended {fs_table}"))
