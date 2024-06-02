# Databricks notebook source
# MAGIC %md ## Basic MLflow training notebooks
# MAGIC
# MAGIC ##### Wine Quality train/predict model notebooks
# MAGIC * Canonical Sklearn wine quality model showing extensive MLflow functionality
# MAGIC   * [Sklearn_Wine_UC]($Sklearn_Wine_UC) - uses Unity Catalog model registry
# MAGIC    * [Sklearn_Wine]($Sklearn_Wine) - uses Workspace model registry
# MAGIC * [Sklearn_Wine_ONNX]($Sklearn_Wine_ONNX) - ONNX with Sklearn version (Sklearn and ONNX models logged)
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
# MAGIC ##### Other basic model notebooks
# MAGIC * [Sklearn_CustomModel]($Sklearn_CustomModel) - three examples of MLflow [Python custom models](https://mlflow.org/docs/latest/models.html#custom-python-models)
# MAGIC * [TensorFlow_MNIST]($TensorFlow_MNIST) - TensorFlow Keras MNIST
# MAGIC
# MAGIC ##### Feature Store notebooks
# MAGIC * Sklearn Wine Quality train/predict model notebooks
# MAGIC * [feature_store_ws]($feature_store/_README) - Uses [FeatureStoreClient](https://api-docs.databricks.com/python/feature-store/latest/feature_store.client.html) - Workspace feature store
# MAGIC * [feature_store_uc]($feature_store_uc/_README) - Uses [FeatureEngineeringClient](https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html) - Unity Catalog feature store
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
# MAGIC Last updated: _2024-06-02_
