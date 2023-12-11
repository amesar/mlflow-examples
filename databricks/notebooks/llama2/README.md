# Take a Llama 2 model for a spin

### Overview

* Example of Llama 2 models available at [Databricks Marketplace Llama2](https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models).
* Demonstrates how to do both real-time and batch model inference.
* GPU cluster configuration:
  * For batch, use `g4dn.xlarge` instance type (AWS).
  * For model serving use:
    * `Workload type` - GPU_MEDIUM
    * `Workload size` - Small

### Notebooks
* [Batch_Score_Llama_2](Batch_Score_Llama_2.py) - Batch scoring with Spark UDF.
  * Input questions can be read from a table or file and output can be stored in a table.
* [Model_Serve_Llama_2](Model_Serve_Llama_2.py) - Real-time scoring with model serving endpoint.
* [Common](Common.py)

### Enhanced Databricks Marketplace Sample
* Simpler example demonstrating both batch a real-time inference. 
* Enhanced with parameterized widgets to replaced hard-coded values. 
* [Llama 2 Marketplace Listing Sample](Llama_2_Marketplace_Listing_Sample.py) - enhanced notebook of below.
* [llama_2_marketplace_listing_example](https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models) - Original Marketplace example notebook.

### Github
* https://github.com/amesar/mlflow-examples/tree/master/databricks/notebooks/llama2

#### Last updated: _2023-12-10_
