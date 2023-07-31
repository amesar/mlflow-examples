# Databricks notebook source
# MAGIC %md ## Text to Text Generation Task
# MAGIC
# MAGIC ##### Hugging Face model 
# MAGIC * Model: [declare-lab/flan-alpaca-base](https://huggingface.co/declare-lab/flan-alpaca-base)
# MAGIC * Task: text2text-generation 
# MAGIC * Description: Text-To-Text Generation
# MAGIC
# MAGIC ##### Related Hugging Face models
# MAGIC * [declare-lab/flan-alpaca-base](https://huggingface.co/declare-lab/flan-alpaca-base)
# MAGIC * [declare-lab/flan-alpaca-large](https://huggingface.co/declare-lab/flan-alpaca-large)
# MAGIC
# MAGIC ##### Sample registered model names
# MAGIC * examples_transfors_simple
# MAGIC * andre_catalog.llm_models.examples_transfors_simple
# MAGIC
# MAGIC ##### MLflow `transformers` flaver
# MAGIC ```
# MAGIC {
# MAGIC   "transformers_version": "4.28.1",
# MAGIC   "code": null,
# MAGIC   "task": "text2text-generation",
# MAGIC   "instance_type": "Text2TextGenerationPipeline",
# MAGIC   "source_model_name": "declare-lab/flan-alpaca-base",
# MAGIC   "pipeline_model_type": "T5ForConditionalGeneration",
# MAGIC   "framework": "pt",
# MAGIC   "tokenizer_type": "T5TokenizerFast",
# MAGIC   "components": [ "tokenizer" ],
# MAGIC   "model_binary": "model"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC #### Based upon
# MAGIC * [github.com/mlflow/mlflow/examples/transformers/simple.py](https://github.com/mlflow/mlflow/blob/master/examples/transformers/simple.py)

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
registered_model_name = dbutils.widgets.get("1. Registered model")

dbutils.widgets.text("2. Run name", "")
run_name = dbutils.widgets.get("2. Run name")
run_name = None if run_name=="" else run_name

dbutils.widgets.text("3. Transformer", "declare-lab/flan-alpaca-base")
hf_model_name = dbutils.widgets.get("3. Transformer")

print("registered_model_name:", registered_model_name)
print("hf_model_name:", hf_model_name)
print("run_name:", run_name)

# COMMAND ----------

client = get_client(registered_model_name)
client._registry_uri

# COMMAND ----------

# MAGIC %md ### Setup transformer 

# COMMAND ----------

task = "text2text-generation"
inference_config = {"max_length": 512, "do_sample": True}

task, inference_config

# COMMAND ----------

import transformers

generation_pipeline = transformers.pipeline(
    task = task,
    model = hf_model_name,
)

# COMMAND ----------

# MAGIC %md ### Log model 

# COMMAND ----------

input_example = ["prompt 1", "prompt 2", "prompt 3"]

signature = mlflow.models.infer_signature(
    input_example,
    mlflow.transformers.generate_signature_output(generation_pipeline, input_example)
)

# COMMAND ----------

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(inference_config)
    model_info = mlflow.transformers.log_model(
        transformers_model=generation_pipeline,
        artifact_path="text_generator",
        inference_config=inference_config,
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md ### Show ModelInfo

# COMMAND ----------

dump_obj(model_info)

# COMMAND ----------

dump_flavor(model_info)

# COMMAND ----------

# MAGIC %md ### Add transformer flavor as run tags

# COMMAND ----------

hf_tags = add_transformer_tags(client, model_info)
hf_tags

# COMMAND ----------

# MAGIC %md ### Register model

# COMMAND ----------

version = create_model_version(client, registered_model_name, model_info.artifact_path, run, hf_tags)

# COMMAND ----------

dump_obj(version)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)
sentence_generator

# COMMAND ----------

print(
    sentence_generator.predict(
        ["tell me a story about rocks", 
         "Tell me a joke about a dog that likes spaghetti"]
    )
)

# COMMAND ----------

print(
    sentence_generator.predict(
        ["tell me a story about climbing k2 mountain"]
    )
)

# COMMAND ----------

# MAGIC %md ### Return

# COMMAND ----------

dbutils.notebook.exit(create_results(model_info, version))
