# Databricks notebook source
# MAGIC %md ### Work - Search_Model_Versions
# MAGIC
# MAGIC ##### Models
# MAGIC * examples_transfors_simple
# MAGIC * andre_catalog.llm_models.examples_transfors_simple
# MAGIC

# COMMAND ----------

# MAGIC %md ### Issues
# MAGIC
# MAGIC
# MAGIC ### UC
# MAGIC
# MAGIC ##### 1. search_model_versions
# MAGIC
# MAGIC tags are not returned if they exist
# MAGIC
# MAGIC ##### 2. search_model_versions - filter with tag
# MAGIC
# MAGIC Returns wrong error message
# MAGIC
# MAGIC Filter
# MAGIC ```
# MAGIC "name='andre_catalog.transformer_models.Text_to_Text_Generation_Task' and tags.hf_source_model_name='declare-lab/flan-alpaca-base'"
# MAGIC ```
# MAGIC
# MAGIC Error message
# MAGIC ```
# MAGIC RestException: INVALID_PARAMETER_VALUE: 
# MAGIC Bad model name: please specify all three levels of the model in the form `catalog_name.schema_name.model_name`
# MAGIC ```
# MAGIC
# MAGIC ##### 2. search_model_versions - filter with like
# MAGIC
# MAGIC Filter
# MAGIC ```
# MAGIC "name like '%_Task'"
# MAGIC ```
# MAGIC
# MAGIC Error message
# MAGIC ```
# MAGIC RestException: INVALID_PARAMETER_VALUE: 
# MAGIC Unsupported filter query : `name like '%_Task'`. Please specify your filter parameter in the format `name = 'model_name'`
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

# XX

model_name = "examples_transfors_simple"
model_name = "andre_catalog.llm_models.examples_transfors_simple"

model_name = "andre_catalog.transformer_models.Text_to_Text_Generation_Task"
version = "1"

#model_name = "Text_to_Text_Generation_Task"
version = "1"

model_name

# COMMAND ----------

client = get_client(model_name)
client._registry_uri

# COMMAND ----------

# MAGIC %md ### Get model version

# COMMAND ----------

client.set_model_version_tag(vr.name, vr.version, "foo", "bar")

# COMMAND ----------

vr = client.get_model_version(model_name, version)
dump_obj(vr)

# COMMAND ----------

dump_dict_as_json(vr.tags)

# COMMAND ----------

# MAGIC %md ### Search 1 - model name

# COMMAND ----------

filter = f"name='{model_name}'"
filter

# COMMAND ----------

vrs = client.search_model_versions(filter)
len(vrs)

# COMMAND ----------

for vr in vrs:
    dump_obj(vr)

# COMMAND ----------

# MAGIC %md ### Search 2 - model name with like - XXX

# COMMAND ----------

##filter = f"name like '{model_name}_%'"
filter = f"name like '%_Task'"

filter

# COMMAND ----------

vrs = client.search_model_versions(filter)
len(vrs)

# COMMAND ----------

# MAGIC %md ### Search 3 - model name and tag

# COMMAND ----------

# 'declare-lab/flan-alpaca-base'

# COMMAND ----------

filter = f"name='{model_name}' and tags.hf_source_model_name='declare-lab/flan-alpaca-base'"
#filter = f"name='{model_name}'"
filter

# COMMAND ----------

vrs = client.search_model_versions(filter)
len(vrs)

# COMMAND ----------

for vr in vrs:
    dump_obj(vr)
