# Databricks notebook source
# MAGIC %md ## Conversational Task
# MAGIC
# MAGIC ##### Hugging Face model 
# MAGIC * Task: conversational
# MAGIC * Model: [microsoft/DialoGPT-medium ](https://huggingface.co/microsoft/DialoGPT-medium)
# MAGIC * Description: Pretrained dialogue response generation model for multiturn conversations
# MAGIC
# MAGIC ##### MLflow `transformers` flaver
# MAGIC ```
# MAGIC {
# MAGIC   "transformers_version": "4.28.1",
# MAGIC   "code": null,
# MAGIC   "task": "conversational",
# MAGIC   "instance_type": "ConversationalPipeline",
# MAGIC   "source_model_name": "microsoft/DialoGPT-medium",
# MAGIC   "pipeline_model_type": "GPT2LMHeadModel",
# MAGIC   "framework": "pt",
# MAGIC   "tokenizer_type": "GPT2TokenizerFast",
# MAGIC   "components": [
# MAGIC     "tokenizer"
# MAGIC   ],
# MAGIC   "model_binary": "model"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC #### Based upon
# MAGIC * [github.com/mlflow/mlflow/examples/transformers/conversational.py](https://github.com/mlflow/mlflow/blob/master/examples/transformers/conversational.py)

# COMMAND ----------

# MAGIC %md ### Setup  

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
registered_model_name = dbutils.widgets.get("1. Registered model")
registered_model_name

# COMMAND ----------

# MAGIC %md ### Setup transformer 

# COMMAND ----------

import transformers
import mlflow

conversational_pipeline = transformers.pipeline(model="microsoft/DialoGPT-medium")

signature = mlflow.models.infer_signature(
    "Hi there, chatbot!",
    mlflow.transformers.generate_signature_output(conversational_pipeline, "Hi there, chatbot!"),
)

# COMMAND ----------

# MAGIC %md ### Log Model

# COMMAND ----------

with mlflow.start_run() as run:
    model_info = mlflow.transformers.log_model(
        transformers_model=conversational_pipeline,
        artifact_path="chatbot",
        task="conversational",
        signature=signature,
        input_example="A clever and witty question",
    )

# COMMAND ----------

# MAGIC %md ### Register model

# COMMAND ----------

version = register_model(client, registered_model_name, model_info, run)

# COMMAND ----------

dump_obj(version)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

# Load the conversational pipeline as an interactive chatbot

chatbot = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
chatbot

# COMMAND ----------

first = chatbot.predict("What is the best way to get to Antarctica?")
print(f"Response: {first}")

# COMMAND ----------

second = chatbot.predict("What kind of boat should I use?")
print(f"Response: {second}")

# COMMAND ----------

# MAGIC %md ### Return

# COMMAND ----------

dbutils.notebook.exit(create_results(model_info, version))
