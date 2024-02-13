# Databricks notebook source
# MAGIC %md ## Phi2
# MAGIC
# MAGIC ##### Hugging Face model 
# MAGIC * Task: translation_en_to_fr
# MAGIC * Model: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
# MAGIC * Description:
# MAGIC
# MAGIC ##### MLflow `transformers` flavor
# MAGIC ```
# MAGIC TODO
# MAGIC ```
# MAGIC
# MAGIC #### Based upon
# MAGIC * [github.com/mlflow/mlflow/examples/transformers/sentence_transformer.py](https://github.com/mlflow/mlflow/blob/master/examples/transformers/sentence_transformer.py)
# MAGIC

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
registered_model_name = dbutils.widgets.get("1. Registered model")
registered_model_name

# COMMAND ----------

# MAGIC %md #### Create model

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# COMMAND ----------

# MAGIC %md #### Create signature

# COMMAND ----------

signature = mlflow.models.infer_signature(
    "Hi there, chatbot!",
    mlflow.transformers.generate_signature_output(model, "Hi there, chatbot!"),
)

# COMMAND ----------

# MAGIC %md #### Log Model

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

# MAGIC %md #### Add transformer flavor as run tags

# COMMAND ----------

dump_flavor(model_info)

# COMMAND ----------

add_transformer_tags(client, model_info)

# COMMAND ----------

# MAGIC %md #### Predict

# COMMAND ----------

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', 
   return_tensors="pt", 
   return_attention_mask=False
)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
