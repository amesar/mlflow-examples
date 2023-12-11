# Databricks notebook source
# MAGIC %md ### Take Llama 2 model for a spin
# MAGIC
# MAGIC * Simple example for Llama 2 models available at [Databricks Marketplace Llama2](https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models).
# MAGIC   * See [llama_2_marketplace_listing_example](https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models) Marketplace example notebook.
# MAGIC * Demonstrates how to do both real-time and batch model inference.
# MAGIC * Cluster instance type:
# MAGIC   * For batch, use `g4dn.xlarge`.
# MAGIC   * For model serving, use GPU_MEDIUM for `Workload type` and Small for `Workload size`.
# MAGIC
# MAGIC ##### Notebooks
# MAGIC * [Batch_Score_Llama_2]($Batch_Score_Llama_2) - Batch scoring with Spark UDF.
# MAGIC * [Model_Serve_Llama_2]($Model_Serve_Llama_2) - Real-time scoring with model serving endpoint.
# MAGIC * [Common]($Common)
# MAGIC
# MAGIC ##### Github
# MAGIC * https://github.com/amesar/mlflow-examples/tree/master/databricks/notebooks/llama2
# MAGIC
# MAGIC
# MAGIC ##### Last updated: _2023-12-10_
