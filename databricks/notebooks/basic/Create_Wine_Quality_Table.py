# Databricks notebook source
# MAGIC %md ## Create Wine Quality Table
# MAGIC
# MAGIC Create `winequality_white` table for Sklearn_Wine notebooks.
# MAGIC
# MAGIC ##### Widgets
# MAGIC * `Data path` - Path to DBFS CSV data.
# MAGIC * `Table`:
# MAGIC   * UC - `uc_andre.ml_data.winequality_white`
# MAGIC   * non-UC - `andre.winequality_white`
# MAGIC
# MAGIC Last updated: 2023-09-12

# COMMAND ----------

dbutils.widgets.text("Data path", "dbfs:/databricks-datasets/wine-quality/winequality-white.csv")
dbutils.widgets.text("Table", "")
table_name = dbutils.widgets.get("Table")
data_path = dbutils.widgets.get("Data path")

print("table_name:", table_name)
print("data_path:", data_path)

# COMMAND ----------

if not table_name: raise Exception(f"Missing widget 'Table'")

# COMMAND ----------

idx = table_name.rfind(".")
database_name = table_name[:idx]
database_name

# COMMAND ----------

# MAGIC %md #### Database

# COMMAND ----------

from pyspark.sql.functions import *

df = spark.sql("show databases").select("databaseName").filter(col("databaseName").like("andre_%"))
display(df)

# COMMAND ----------

spark.sql(f"create database if not exists {database_name}")

# COMMAND ----------

# MAGIC %md #### Table

# COMMAND ----------

import os
os.environ["DATA_PATH"] = data_path.replace("dbfs:","/dbfs")

# COMMAND ----------

# MAGIC %sh ls -l $DATA_PATH

# COMMAND ----------

df = (spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .option("delimiter",";")
    .load(data_path))

# COMMAND ----------

columns = [ col.replace(" ","_") for col in df.columns ]
df = df.toDF(*columns)
display(df)

# COMMAND ----------

df.write.mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

df = spark.table(table_name)
display(df)
