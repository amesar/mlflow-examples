# Databricks notebook source
# MAGIC %md ### Take Llama 2 model for a spin
# MAGIC
# MAGIC * Example for Llama 2 models available at [Databricks Marketplace Llama2](https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models).
# MAGIC * Demonstrates how to do both real-time and batch model inference.
# MAGIC * GPU cluster configuration:
# MAGIC   * For batch, use `g4dn.xlarge` instance type (AWS).
# MAGIC   * For model serving use:
# MAGIC     * `Workload type` - GPU_MEDIUM
# MAGIC     * `Workload size` - Small
# MAGIC
# MAGIC ##### Notebooks
# MAGIC * [Batch_Score_Llama_2]($Batch_Score_Llama_2) - Batch scoring with Spark UDF.
# MAGIC   * Input questions can be read from a table or file and output can be stored in a table.
# MAGIC * [Model_Serve_Llama_2]($Model_Serve_Llama_2) - Real-time scoring with model serving endpoint.
# MAGIC * [Common]($Common)
# MAGIC
# MAGIC ##### Enhanced Databricks Marketplace Sample
# MAGIC * Simpler example demonstrating both batch a real-time inference. 
# MAGIC * Enhanced with parameterized widgets to replaced hard-coded values. 
# MAGIC * [Llama 2 Marketplace Listing Sample]($Llama_2_Marketplace_Listing_Sample) - enhanced notebook of below.
# MAGIC * [llama_2_marketplace_listing_example](https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models) - Original Marketplace example notebook.
# MAGIC
# MAGIC ##### Github
# MAGIC * https://github.com/amesar/mlflow-examples/tree/master/databricks/notebooks/llama2
# MAGIC
# MAGIC ##### Last updated: _2023-12-10_
