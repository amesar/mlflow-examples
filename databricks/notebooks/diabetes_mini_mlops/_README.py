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
# MAGIC ##### Github
# MAGIC * https://github.com/amesar/mlflow-examples/tree/master/databricks/notebooks/diabetes_mini_mlops
# MAGIC
# MAGIC ##### Last updated: _2023-12-13_

# COMMAND ----------

# MAGIC %md ### Mini MLOps Pipeline diagram

# COMMAND ----------

# MAGIC %md 
# MAGIC **Batch Scoring Pipeline**
# MAGIC
# MAGIC <img src="https://github.com/amesar/mlflow-examples/blob/master/python/e2e-ml-pipeline/e2e_ml_batch_pipeline.png?raw=true"  width="450" />
# MAGIC
# MAGIC **Real-time Scoring Pipeline**
# MAGIC
# MAGIC <img src="https://github.com/amesar/mlflow-examples/blob/master/python/e2e-ml-pipeline/e2e_ml_realtime_pipeline.png?raw=true"  width="700" />

# COMMAND ----------

# MAGIC %md ### Advanced MLOps Pipeline diagram

# COMMAND ----------

# MAGIC %md 
# MAGIC <img src="https://github.com/amesar/mlflow-resources/blob/master/images/databricks/mlops_pipeline_uc.png?raw=true"  width="900" />
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC 1. Dev environment
# MAGIC     1a. Train model
# MAGIC     1b. Register best model in dev catalog
# MAGIC     1b. Copy MLflow run to staging workspace
# MAGIC 2. Staging environment
# MAGIC     2a. Run model evaluation and non-ML code tests
# MAGIC     2b. Copy (promote) model version to staging catalog
# MAGIC     2c. Copy model version to prod catalog when ready
# MAGIC     2c. Copy MLflow run to prod workspace for lineage and governance
# MAGIC 3. Prod environment - run model inference on data
# MAGIC ```
