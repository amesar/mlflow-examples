# Databricks notebook source
# MAGIC %md ## Run Smoke Test for Notebooks

# COMMAND ----------

notebook = "01_Train_Model"
result = dbutils.notebook.run(notebook, 600, {} )
result

# COMMAND ----------

notebook = "02_Register_Model"
result = dbutils.notebook.run(notebook, 600, {} )
result

# COMMAND ----------

notebook = "01_Train_Model"
result = dbutils.notebook.run(notebook, 600, {})
result
