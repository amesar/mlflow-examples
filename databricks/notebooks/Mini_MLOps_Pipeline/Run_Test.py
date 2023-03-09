# Databricks notebook source
# MAGIC %md ### Run Tests

# COMMAND ----------

notebook = "01_Train_Model"
result = dbutils.notebook.run(notebook, 600, {} )
result

# COMMAND ----------

