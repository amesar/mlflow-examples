# Databricks notebook source
# MAGIC %md # MLflow MLOps End-to-End Pipeline Tutorial
# MAGIC 
# MAGIC **Overview**
# MAGIC 
# MAGIC Basic tutorial demonstrating an end end-to-end MLOps pipeline using MLflow: 
# MAGIC * Trains several models with different hyperparameter values.
# MAGIC * Shows different ways to view and search model runs.
# MAGIC * Promotes best model to the model registry as a production stage.
# MAGIC * There are two types of model scoring:
# MAGIC   * Batch scoring.
# MAGIC   * Real-time scoring with Databricks model server.
# MAGIC 
# MAGIC **Batch Scoring Pipeline**
# MAGIC 
# MAGIC <img src="https://github.com/amesar/mlflow-examples/blob/master/python/e2e-ml-pipeline/e2e_ml_batch_pipeline.png?raw=true"  width="450" />
# MAGIC 
# MAGIC **Real-time Scoring Pipeline**
# MAGIC 
# MAGIC <img src="https://github.com/amesar/mlflow-examples/blob/master/python/e2e-ml-pipeline/e2e_ml_realtime_pipeline.png?raw=true"  width="700" />
# MAGIC 
# MAGIC **Notebooks**
# MAGIC * [01_Train_Model]($01_Train_Model) - train a Sklearn model several times with different `max_depth` hyperparameter values using the wine quality dataset.
# MAGIC   * Option to use [MLflow Autologging](https://docs.databricks.com/applications/mlflow/databricks-autologging.html)  or explicit MLflow API calls.
# MAGIC * [02_Search_Model_Runs]($02_Search_Model_Runs) - show different ways to view and search runs.
# MAGIC * [03_Register_Model]($03_Register_Model) - register the best model in the model registry.
# MAGIC   * Find the best run by searching for the run with the lowest RMSE metric.
# MAGIC   * Add this best run's model to Model Registry and promote it to the `production` stage as `MLflow_MLOps_E2E_pipeline/production`.
# MAGIC * Score model 
# MAGIC   * [04_Model_Predict_Batch]($04_Model_Predict_Batch) - score in batch mode.
# MAGIC     * Load `models:/MLflow MLOps E2E Pipeline/production` from the Model Registry and score as Sklearn, PyFunc and UDF flavors.
# MAGIC   * [05_Model_Predict_Realtime]($05_Model_Predict_Realtime) - score in realtime against a model serving cluster.
# MAGIC     * Starts a Databricks model serving cluster and then submits predictions with curl.
# MAGIC * [Common]($Common) - common utilities.
# MAGIC 
# MAGIC **Setup**
# MAGIC * Use the latest Databricks ML Runtime cluster.
# MAGIC   
# MAGIC Last updated: 2022-09-01