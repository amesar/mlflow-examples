# Databricks notebook source
# MAGIC %md ## Basic MLflow training notebooks
# MAGIC
# MAGIC ##### Overview
# MAGIC * The goal of these notebooks is to demonstrate basic and advanced MLflow functionality.
# MAGIC * The focus is on showing many of MLflow features and the complexity of the models themselves is kept to a minimum.
# MAGIC * These notebooks are for "classical" non-LLM MLflow. 
# MAGIC
# MAGIC ##### Wine Quality train/predict model notebooks
# MAGIC * Canonical Sklearn wine quality model showing extensive MLflow functionality
# MAGIC   * [Sklearn_Wine_UC]($Sklearn_Wine_UC) - uses Unity Catalog model registry
# MAGIC    * [Sklearn_Wine]($Sklearn_Wine) - uses Workspace model registry
# MAGIC * [Sklearn_Wine_ONNX]($Sklearn_Wine_ONNX) - logs both Sklearn and ONNX models in the run
# MAGIC * [XGBoost_Wine]($XGBoost_Wine) -  XGBoost/Sklearn wine quality model
# MAGIC * [SparkML_Wine]($SparkML_Wine) - SparkML wine quality model
# MAGIC
# MAGIC ##### Wine Quality predict-only model notebooks
# MAGIC * [Predict_Sklearn_Wine]($Predict_Sklearn_Wine)
# MAGIC * [Predict_ONNX_Wine]($Predict_ONNX_Wine)
# MAGIC
# MAGIC ##### Sklearn Iris model notebooks
# MAGIC * [Sklearn_Iris]($Sklearn_Iris) 
# MAGIC * [Sklearn_Iris_Autolog]($Sklearn_Iris_Autolog) 
# MAGIC
# MAGIC ##### Custom Python model notebooks
# MAGIC * [Sklearn_CustomModel]($Sklearn_CustomModel) - three simple examples of MLflow [Custom Python models](https://mlflow.org/docs/latest/models.html#custom-python-models)
# MAGIC * [Custom_MultiModel_Run]($Custom_MultiModel_Run) - Multimodel example
# MAGIC
# MAGIC ##### Other model notebooks
# MAGIC * [TensorFlow_MNIST]($TensorFlow_MNIST) - TensorFlow Keras MNIST
# MAGIC * [Nested_Runs_Example]($Nested_Runs_Example) 
# MAGIC
# MAGIC ##### Feature Store notebooks
# MAGIC * [feature_store_uc]($feature_store_uc/_README) - Uses [FeatureEngineeringClient](https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html) - Unity Catalog feature store
# MAGIC * [feature_store_ws]($feature_store/_README) - Uses [FeatureStoreClient](https://api-docs.databricks.com/python/feature-store/latest/feature_store.client.html) - Workspace feature store
# MAGIC
# MAGIC ##### Helper notebooks
# MAGIC * [Common]($Common) - Shared logic.
# MAGIC * [Versions]($Versions) - Show version information.
# MAGIC * [Create_Wine_Quality_Table]($Create_Wine_Quality_Table) - Create winequality_white table for Sklearn_Wine notebooks.
# MAGIC
# MAGIC ##### Scala SparkML Interop Notebooks
# MAGIC * [README]($scala_sparkml_interop/00_README)
# MAGIC
# MAGIC ##### Github
# MAGIC * https://github.com/amesar/mlflow-examples/tree/master/databricks/notebooks/basic
# MAGIC
# MAGIC Last updated: _2024-07-21_
