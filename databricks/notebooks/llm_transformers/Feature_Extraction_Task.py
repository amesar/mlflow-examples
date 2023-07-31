# Databricks notebook source
# MAGIC %md ## Feature Extraction Task
# MAGIC
# MAGIC ##### Hugging Face model 
# MAGIC * Task: feature-extraction
# MAGIC * Model: [sentence-transformers/all-MiniLM-L12-v2 ](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
# MAGIC * Description: TODO
# MAGIC
# MAGIC ##### MLflow `transformers` flaver
# MAGIC ```
# MAGIC {
# MAGIC   "transformers_version": "4.28.1",
# MAGIC   "code": null,
# MAGIC   "task": "feature-extraction",
# MAGIC   "instance_type": "FeatureExtractionPipeline",
# MAGIC   "source_model_name": "sentence-transformers/all-MiniLM-L12-v2",
# MAGIC   "pipeline_model_type": "BertModel",
# MAGIC   "framework": "pt",
# MAGIC   "tokenizer_type": "BertTokenizerFast",
# MAGIC   "components": [
# MAGIC     "tokenizer"
# MAGIC   ],
# MAGIC   "model_binary": "model"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC #### Based upon
# MAGIC * [github.com/mlflow/mlflow/examples/transformers/sentence_transformer.py](https://github.com/mlflow/mlflow/blob/master/examples/transformers/sentence_transformer.py)
# MAGIC
# MAGIC
# MAGIC _Last updated: 2023-07-29_

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
registered_model_name = dbutils.widgets.get("1. Registered model")
registered_model_name

# COMMAND ----------

client = get_client(registered_model_name)

# COMMAND ----------

# MAGIC %md ### Log Model

# COMMAND ----------

import torch
from transformers import BertModel, BertTokenizerFast, pipeline
import mlflow


sentence_transformers_architecture = "sentence-transformers/all-MiniLM-L12-v2"
task = "feature-extraction"

model = BertModel.from_pretrained(sentence_transformers_architecture)
tokenizer = BertTokenizerFast.from_pretrained(sentence_transformers_architecture)

sentence_transformer_pipeline = pipeline(task=task, model=model, tokenizer=tokenizer)

with mlflow.start_run() as run:
    model_info = mlflow.transformers.log_model(
        transformers_model=sentence_transformer_pipeline,
        artifact_path="sentence_transformer",
        framework="pt",
        torch_dtype=torch.bfloat16,
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

def pool_and_normalize_encodings(input_sentences, model, tokenizer, **kwargs):
    def pool(model_output, attention_mask):
        embeddings = model_output[0]
        expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * expanded_mask, 1) / torch.clamp(
            expanded_mask.sum(1), min=1e-9
        )

    encoded = tokenizer(
        input_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        model_output = model(**encoded)

    pooled = pool(model_output, encoded["attention_mask"])
    return torch.nn.functional.normalize(pooled, p=2, dim=1)

# COMMAND ----------

sentences = [
    "He said that he's sinking all of his investment budget into coconuts.",
    "No matter how deep you dig, there's going to be a point when it just gets too hot.",
    "She said that there isn't a noticeable difference between a 10 year and a 15 year whisky.",
]

# COMMAND ----------

loaded_model = mlflow.transformers.load_model(model_info.model_uri, return_type="components")
encoded_sentences = pool_and_normalize_encodings(sentences, **loaded_model)
print(encoded_sentences)

# COMMAND ----------

# MAGIC %md ### Return

# COMMAND ----------

dbutils.notebook.exit(create_results(model_info, version))
