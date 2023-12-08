# Databricks notebook source
# MAGIC %md ## Diabetes MLflow Mini MLOps Example
# MAGIC
# MAGIC ##### Overview
# MAGIC
# MAGIC * Trains several model runs with different hyperparameters.
# MAGIC * Registers the best run's model in the Unity Catalog model registry with the 'champ' alias.
# MAGIC * Scores with either:
# MAGIC   * Batch scoring with Spark.
# MAGIC   * Real-time model scoring with Serverless Model Serving ([AWS](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-real-time-inference.html) - [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)).
# MAGIC
# MAGIC ##### Notebooks
# MAGIC
# MAGIC * [01_Train_Model]($01_Train_Model) - Run several Sklearn training runs with different hyperparameters.
# MAGIC * [02_Register_Model]($02_Register_Model) - Find the best run and register it as model version.
# MAGIC * [03_Batch_Scoring]($03_Batch_Scoring) - Batch score with Sklearn, Pyfunc and UDF flavors.
# MAGIC * Real-time Model Serving endpoint - using [Serverless Model Serving](https://docs.databricks.com/machine-learning/model-inference/serverless/serverless-real-time-inference.html).
# MAGIC   * [04a_Model_Serving_Start]($04a_Model_Serving_Start) - Start endpoint.
# MAGIC   * [04b_Model_Serving_Score]($04b_Model_Serving_Score) - Score endpoint.
# MAGIC   * [04c_Model_Serving_Stop]($04c_Model_Serving_Stop) - Stop endpoint.
# MAGIC * Helper notebooks:
# MAGIC   * [Common]($includes/Common) - Helper functions.
# MAGIC   * [ModelServingClient]($../includes/ModelServingClient) - Python HTTP client to invoke Databricks Model Serving API.
# MAGIC   * [HttpClient]($../includes/HttpClient) - Python HTTP client to invoke Databricks API.
# MAGIC
# MAGIC Last updated: _2023-12-07_

# COMMAND ----------

# MAGIC %md 
# MAGIC **Batch Scoring Pipeline**
# MAGIC
# MAGIC <img src="https://github.com/amesar/mlflow-examples/blob/master/python/e2e-ml-pipeline/e2e_ml_batch_pipeline.png?raw=true"  width="450" />
# MAGIC
# MAGIC **Real-time Scoring Pipeline**
# MAGIC
# MAGIC <img src="https://github.com/amesar/mlflow-examples/blob/master/python/e2e-ml-pipeline/e2e_ml_realtime_pipeline.png?raw=true"  width="700" />