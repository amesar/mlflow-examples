# Databricks notebook source
# MAGIC %md
# MAGIC # Overview of llama_2 models in Databricks Marketplace Listing
# MAGIC
# MAGIC Enhanced version (parameterized with widgets) of the Datarbicks Marketplace 
# MAGIC [llama_2_marketplace_listing_example](https://marketplace.databricks.com/details/46527194-66f5-4ea1-9619-d7ec6be6a2fa/Databricks_Llama-2-Models) notebook.
# MAGIC
# MAGIC The llama_2 models offered in Databricks Marketplace are text generation models released by Meta AI. They are [MLflow](https://mlflow.org/docs/latest/index.html) models that packages
# MAGIC [Hugging Face’s implementation for llama_2 models](https://huggingface.co/meta-llama)
# MAGIC using the [transformers](https://mlflow.org/docs/latest/models.html#transformers-transformers-experimental)
# MAGIC flavor in MLflow.
# MAGIC
# MAGIC **Input:** string containing the text of instructions
# MAGIC
# MAGIC **Output:** string containing the generated response text
# MAGIC
# MAGIC For example notebooks of using the llama_2 model in various use cases on Databricks, please refer to [the Databricks ML example repository](https://github.com/databricks/databricks-ml-examples/tree/master/llm-models/llamav2).

# COMMAND ----------

# MAGIC %md
# MAGIC # Listed Marketplace Models
# MAGIC - llama_2_7b_chat_hf:
# MAGIC   - It packages [Hugging Face’s implementation for the llama_2_7b_chat_hf model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
# MAGIC   - It has 7 billion parameters.
# MAGIC   - While it offers the fastest processing speed, it may have lower quality compared to other models in the model family.
# MAGIC - llama_2_13b_chat_hf:
# MAGIC   - It packages [Hugging Face’s implementation for the llama_2_13b_chat_hf model](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf).
# MAGIC   - It has 13 billion parameters.
# MAGIC   - It offers a middle ground on speed and quality compared to other models in the model family.
# MAGIC - llama_2_70b_chat_hf:
# MAGIC   - It packages [Hugging Face’s implementation for the llama_2_70b_chat_hf model](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf).
# MAGIC   - It has 70 billion parameters.
# MAGIC   - This model excels in quality among the models in the model family, but it comes at the cost of being the slowest.

# COMMAND ----------

# MAGIC %md
# MAGIC # Suggested environment
# MAGIC Creating and querying serving endpoint don't require specific runtime versions and GPU instance types, but
# MAGIC for batch inference we suggest the following:
# MAGIC
# MAGIC - Databricks Runtime for Machine Learning version 14.2 or greater
# MAGIC - Recommended instance types:
# MAGIC   | Model Name      | Suggested instance type (AWS) | Suggested instance type (AZURE) | Suggested instance type (GCP) |
# MAGIC   | --------------- | ----------------------------- | ------------------------------- | ----------------------------- |
# MAGIC   | `llama_2_7b_chat_hf` | `g5.8xlarge` | `Standard_NV36ads_A10_v5` | `g2-standard-4`|
# MAGIC   | `llama_2_13b_chat_hf` | `g5.12xlarge` | `Standard_NC24ads_A100_v4` | `g2-standard-48` or `a2-ultragpu-1g`|
# MAGIC   | `llama_2_70b_chat_hf` | `p4d.24xlarge` or `g5.48xlarge` | `Standard_NC48ads_A100_v4` | `a2-highgpu-8g` or  `g2-standard-96`|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies
# MAGIC To create and query the model serving endpoint, Databricks recommends to install the newest Databricks SDK for Python.

# COMMAND ----------

# Upgrade to use the newest Databricks SDK
%pip install --upgrade databricks-sdk
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.__version__

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# Default catalog name when installing the model from Databricks Marketplace.
# Replace with the name of the catalog containing this model
catalog_name = "databricks_llama_2_models"
dbutils.widgets.text("1. Catalog", catalog_name)
catalog = dbutils.widgets.get("1. Catalog")

dbutils.widgets.text("2. Schema", "models")
schema = dbutils.widgets.get("2. Schema")

model_names = ['llama_2_7b_chat_hf', 'llama_2_13b_chat_hf', 'llama_2_70b_chat_hf']
dbutils.widgets.dropdown("3. Model", model_names[0], model_names)
model_name = dbutils.widgets.get("3. Model")

dbutils.widgets.text("4. Version", "1")
version = dbutils.widgets.get("4. Version")

dbutils.widgets.text("5. Workload", "GPU_MEDIUM")
workload_type = dbutils.widgets.get("5. Workload")

print("catalog:       ", catalog)
print("schema:        ", schema)
print("model :        ", model_name)
print("version:       ", version)
print("workload_type: ", workload_type)

# COMMAND ----------

model_uc_path = f"{catalog}.{schema}.{model_name}"
endpoint_name = f'andre_{model_name}_marketplace'

print("model_uc_path:    ", model_uc_path)
print("endpoint_name:    ", endpoint_name)


# COMMAND ----------

# MAGIC %md
# MAGIC # Usage

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks recommends that you primarily work with this model via Model Serving
# MAGIC ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints)).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying the model to Model Serving
# MAGIC
# MAGIC You can deploy this model directly to a Databricks Model Serving Endpoint
# MAGIC ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints)).
# MAGIC
# MAGIC Note: Model serving is not supported on GCP. On GCP, Databricks recommends running `Batch inference using Spark`, 
# MAGIC as shown below.
# MAGIC
# MAGIC We recommend the below workload types for each model size:
# MAGIC | Model Name      | Suggested workload type (AWS) | Suggested workload type (AZURE) |
# MAGIC | --------------- | ----------------------------- | ------------------------------- |
# MAGIC | `llama_2_7b_chat_hf` | GPU_MEDIUM | GPU_LARGE |
# MAGIC | `llama_2_13b_chat_hf` | MULTIGPU_MEDIUM | GPU_LARGE |
# MAGIC | `llama_2_70b_chat_hf` | Please reach out to your Databricks representative or [submit the preview enrollment form](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit?ts=65124ee0) to access the necessary GPUs and get information on the required `workload_type` to pass when creating a serving endpoint. ||
# MAGIC
# MAGIC You can create the endpoint by clicking the “Serve this model” button above in the model UI. And you can also
# MAGIC create the endpoint with Databricks SDK as following:
# MAGIC

# COMMAND ----------

import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
w = WorkspaceClient()

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": endpoint_name,
            "model_name": model_uc_path,
            "model_version": version,
            "workload_type": workload_type,
            "workload_size": "Small",
            "scale_to_zero_enabled": "False",
        }
    ]
})
model_details = w.serving_endpoints.create(name=endpoint_name, config=config)
model_details.result(timeout=datetime.timedelta(minutes=60))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL transcription using ai_query
# MAGIC
# MAGIC To generate the text using the endpoint, use `ai_query`
# MAGIC to query the Model Serving endpoint. The first parameter should be the
# MAGIC name of the endpoint you previously created for Model Serving. The second
# MAGIC parameter should be a `named_struct` with name `prompt` and value is the 
# MAGIC column name that containing the instruction text. Extra parameters can be added
# MAGIC to the named_struct too. For supported parameters, please refer to [MLFlow AI gateway completion routes](https://mlflow.org/docs/latest/gateway/index.html#completions)
# MAGIC The third and fourth parameters set the return type, so that
# MAGIC `ai_query` can properly parse and structure the output text.
# MAGIC
# MAGIC NOTE: `ai_query` is currently in Public Preview. Please sign up at [AI Functions Public Preview enrollment form](https://docs.google.com/forms/d/e/1FAIpQLScVyh5eRioqGwuUVxj9JOiKBAo0-FWi7L3f4QWsKeyldqEw8w/viewform) to try out the new feature.
# MAGIC
# MAGIC ```sql
# MAGIC SELECT 
# MAGIC ai_query(
# MAGIC   <endpoint name>,
# MAGIC   named_struct("prompt", "What is ML?",  "max_tokens", 256),
# MAGIC   'returnType',
# MAGIC   'STRUCT<candidates:ARRAY<STRUCT<text:STRING, metadata:STRUCT<finish_reason:STRING>>>, metadata:STRUCT<input_tokens:float, output_tokens:float, total_tokens:float> >'
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC You can use `ai_query` in this manner to generate text in
# MAGIC SQL queries or notebooks connected to Databricks SQL Pro or Serverless
# MAGIC SQL Endpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the text by querying the serving endpoint
# MAGIC With the Databricks SDK, you can query the serving endpoint as follows:

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Change it to your own input
dataframe_records = [
    {"prompt": "What is ML?", "max_tokens": 512}
]

w = WorkspaceClient()
w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=dataframe_records,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference using Spark
# MAGIC
# MAGIC You can also directly load the model as a Spark UDF and run batch
# MAGIC inference on Databricks compute using Spark. We recommend using a
# MAGIC GPU cluster with Databricks Runtime for Machine Learning version 14.1
# MAGIC or greater.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

logged_model = f"models:/{catalog}.models.{model_name}/{version}"
logged_model

# COMMAND ----------

generate = mlflow.pyfunc.spark_udf(spark, logged_model, "string")

# COMMAND ----------

import pandas as pd

df = spark.createDataFrame(pd.DataFrame({"text": pd.Series("What is ML?")}))

# You can use the UDF directly on a text column
generated_df = df.select(generate(df.text).alias('generated_text'))

# COMMAND ----------

display(generated_df)
