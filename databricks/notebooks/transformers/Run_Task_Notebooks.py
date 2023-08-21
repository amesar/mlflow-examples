# Databricks notebook source
# MAGIC %md ## Run_Task_Notebooks
# MAGIC
# MAGIC Run all the transformer notebooks as a suite.
# MAGIC
# MAGIC **Widgets**
# MAGIC * `1. UC catalog.schema prefix`
# MAGIC   * If empty, register the model in the Workspace Model Registry.
# MAGIC   * If not empty, register the model in the Unity Catalog Model Registry.
# MAGIC     * Sample prefix: `andre_catalog.transformer_models` or `andre_m.transformer_models`.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. UC catalog.schema prefix", "")
uc_prefix = dbutils.widgets.get("1. UC catalog.schema prefix")
uc_prefix

# COMMAND ----------

def run_notebook(notebook):
    model_name = f"{uc_prefix}.{notebook}"if uc_prefix else notebook
    params = { "1. Registered model": model_name }
    print(f"=============")
    #print("params:", params)
    print(f"Registering model '{model_name}' for notebook '{notebook}'")
    result = dbutils.notebook.run(notebook, 600, params )
    print(f"Result for {notebook}: {result}")

# COMMAND ----------

notebooks = [
    "Text_to_Text_Generation_Task",
    "Translation_Task",
    "Conversational_Task",
    "Feature_Extraction_Task",
    "Speech_Recognition_Task"
]

# COMMAND ----------

for nb in notebooks:
    run_notebook(nb)
