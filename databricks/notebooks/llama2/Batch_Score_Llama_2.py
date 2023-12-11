# Databricks notebook source
# MAGIC %md ### Batch Score LLama 2 model
# MAGIC
# MAGIC * Simple Llama2 model batch scoring example.
# MAGIC * Load the new Marketplace `LLama 2 7b` model from Unity Catalog (UC) registry and asks some questions.
# MAGIC * Questions can be from a file or a table.
# MAGIC   * The table has a one string column called `question`.
# MAGIC   * The input file is a one column CSV file with no header.
# MAGIC * You can optionally write the answers to an output table.
# MAGIC * All table names are 3 part UC names such `andre_m.ml_data.llama2_answers`.
# MAGIC * Cluster instance type: for `llama_2_7b_chat_hf`, instance `g4dn.xlarge` (AWS) is OK. 
# MAGIC
# MAGIC ##### Widgets
# MAGIC * `1. Model` - Model name.
# MAGIC   * Default is `marketplace_staging_llama_2_models.models.llama_2_7b_chat_hf` on e2-dogfood.
# MAGIC * `2. Version` - model version.
# MAGIC * `3. Input File or Table` -  Input file or table with questions.
# MAGIC * `4. Output Table` - Output table of answers (includes the original question).
# MAGIC * `5. Write mode` - Write mode for output table. If "none", will not write to the table.
# MAGIC
# MAGIC
# MAGIC ##### Last updated: _2023-12-10_

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

dbutils.widgets.text("1. Model", default_model_name)
dbutils.widgets.text("2. Version", "1")
dbutils.widgets.text("3. Input File or Table", "questions.csv")
dbutils.widgets.text("4. Output Table", "")
dbutils.widgets.dropdown("5. Write mode", "none", ["none", "append","overwrite"])

model_name = dbutils.widgets.get("1. Model")
version = dbutils.widgets.get("2. Version")
input_file_or_table = dbutils.widgets.get("3. Input File or Table")
output_table = dbutils.widgets.get("4. Output Table")
write_mode = dbutils.widgets.get("5. Write mode")

print("model:", model_name)
print("version:", version)
print("input_file_or_table:", input_file_or_table)
print("output_table:", output_table)
print("write_mode:", write_mode)

# COMMAND ----------

assert_widget(model_name, "1. Model name")
assert_widget(version, "2. Version")
assert_widget(input_file_or_table, "3. Input File or Table")

# COMMAND ----------

# MAGIC %md #### Load input data

# COMMAND ----------

# MAGIC %md ##### Load input questions from either a file or table

# COMMAND ----------

df_questions = load_data(input_file_or_table)
display(df_questions)

# COMMAND ----------

# MAGIC %md #### Invoke model with questions

# COMMAND ----------

# MAGIC %md ##### Model URI

# COMMAND ----------

model_uri = f"models:/{model_name}/{version}"
model_uri

# COMMAND ----------

# MAGIC %md ##### Load model as Spark UDF
# MAGIC
# MAGIC This may take a few minutes to load the `llama_2_7b_chat_hf` model.

# COMMAND ----------

udf = mlflow.pyfunc.spark_udf(spark, model_uri, "string")

# COMMAND ----------

# MAGIC %md ##### Call model with questions
# MAGIC
# MAGIC Takes about 20 seconds per question for `llama_2_7b_chat_hf` model.

# COMMAND ----------

df_answers = df_questions.select(udf(df_questions.question).alias("answer"))
display(df_answers)

# COMMAND ----------

# MAGIC %md #### Write results to table

# COMMAND ----------

if output_table and write_mode != "none":
    if write_mode == "overwrite":
        spark.sql(f"drop table if exists {output_table}")
    df_answers.write.mode(write_mode).saveAsTable(output_table) 
