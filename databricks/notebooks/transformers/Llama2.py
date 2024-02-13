# Databricks notebook source
# MAGIC %md ## Llama2
# MAGIC
# MAGIC ##### Hugging Face model 
# MAGIC * Task: translation_en_to_fr
# MAGIC * Model: [t5-small](https://huggingface.co/t5-small)
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
# MAGIC ##### Docs
# MAGIC * [xx](https://huggingface.co/docs/transformers/main/model_doc/llama2)
# MAGIC * [xx](https://www.philschmid.de/llama-2)
# MAGIC

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

# MAGIC %md ### Log Model

# COMMAND ----------

# MAGIC %md ### Add transformer flavor as run tags

# COMMAND ----------

dump_flavor(model_info)

# COMMAND ----------

add_transformer_tags(client, model_info)

# COMMAND ----------

# MAGIC %md ### Predict
