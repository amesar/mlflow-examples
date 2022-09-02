# Databricks notebook source
# MAGIC %md # MLflow Tutorial README
# MAGIC 
# MAGIC **Synopsis**
# MAGIC * Train a SparkML model several times with different `maxDepth` hyperparameters
# MAGIC * Find the best run and register it in the Model Registry as model version `Tutorial_Model/production`
# MAGIC * Load `models:/Tutorial_Model/production` from the Model Registry and score
# MAGIC 
# MAGIC **Overview**
# MAGIC * [01_MLflow_SparkML_Tutorial]($01_MLflow_SparkML_Tutorial) notebook:
# MAGIC   * [Train SparkML]($01_MLflow_SparkML_Tutorial) several times with different hyperparameters
# MAGIC   * Show different ways to retrieve runs
# MAGIC   * Show different ways to search for best run 
# MAGIC   * Score best run using `runs` scheme with:
# MAGIC [Spark ML](https://mlflow.org/docs/latest/python_api/mlflow.spark.html#module-mlflow.spark), 
# MAGIC [Pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and
# MAGIC [MLeap](https://mlflow.org/docs/latest/python_api/mlflow.mleap.html)
# MAGIC *  [02_Model_Registry_Tutorial]($02_Model_Registry_Tutorial) notebook:
# MAGIC   * Leverage registry to load model with `models` scheme and score
# MAGIC 
# MAGIC **Setup**
# MAGIC * DB ML Runtime 6.3 or above
# MAGIC * Install maven jar `ml.combust.mleap:mleap-spark_2.11:0.13.0`
# MAGIC   
# MAGIC 
# MAGIC 
# MAGIC Last updated: 2020-03-16

# COMMAND ----------

# MAGIC %md ### Notebooks
# MAGIC * [01_MLflow_SparkML_Train_Tutorial]($01_MLflow_SparkML_Train_Tutorial)
# MAGIC * [02_Model_Registry_Score_Tutorial]($02_Model_Registry_Score_Tutorial)
# MAGIC * [Common]($Common)