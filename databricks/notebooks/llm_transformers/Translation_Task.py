# Databricks notebook source
# MAGIC %md ## Translation Task
# MAGIC
# MAGIC ##### Hugging Face model 
# MAGIC * Task: translation_en_to_fr
# MAGIC * Model: [t5-small](https://huggingface.co/t5-small)
# MAGIC * Description: Text-To-Text Transfer Transformer 
# MAGIC
# MAGIC ##### MLflow `transformers` flaver
# MAGIC ```
# MAGIC {
# MAGIC   "transformers_version": "4.28.1",
# MAGIC   "code": null,
# MAGIC   "task": "translation_en_to_fr",
# MAGIC   "instance_type": "TranslationPipeline",
# MAGIC   "source_model_name": "t5-small",
# MAGIC   "pipeline_model_type": "T5ForConditionalGeneration",
# MAGIC   "framework": "pt",
# MAGIC   "tokenizer_type": "T5TokenizerFast",
# MAGIC   "components": [
# MAGIC     "tokenizer"
# MAGIC   ],
# MAGIC   "model_binary": "model"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC #### Based upon
# MAGIC * [github.com/mlflow/mlflow/examples/transformers/load_components.py](https://github.com/mlflow/mlflow/blob/master/examples/transformers/load_components.py)
# MAGIC

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
registered_model_name = dbutils.widgets.get("1. Registered model")
registered_model_name

# COMMAND ----------

# MAGIC %md ### Log Model

# COMMAND ----------

import transformers
import mlflow

translation_pipeline = transformers.pipeline(
    task="translation_en_to_fr",
    model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
    tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
)

signature = mlflow.models.infer_signature(
    "Hi there, chatbot!",
    mlflow.transformers.generate_signature_output(translation_pipeline, "Hi there, chatbot!"),
)

# COMMAND ----------

with mlflow.start_run() as run:
    model_info = mlflow.transformers.log_model(
        transformers_model=translation_pipeline,
        artifact_path="french_translator",
        signature=signature,
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

translation_components = mlflow.transformers.load_model(
    model_info.model_uri, return_type="components"
)

for key, value in translation_components.items():
    print(f"{key} -> {type(value).__name__}")

response = translation_pipeline("MLflow is great!")

print(response)

# COMMAND ----------

reconstructed_pipeline = transformers.pipeline(**translation_components)

reconstructed_response = reconstructed_pipeline(
    "Transformers make using Deep Learning models easy and fun!"
)

print(reconstructed_response)

# COMMAND ----------

# MAGIC %md ### Return

# COMMAND ----------

dbutils.notebook.exit(create_results(model_info, version))
