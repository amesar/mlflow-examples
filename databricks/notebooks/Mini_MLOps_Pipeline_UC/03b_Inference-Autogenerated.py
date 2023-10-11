# Databricks notebook source
# MAGIC %md
# MAGIC This is an auto-generated notebook to perform batch inference on a Spark DataFrame using a selected model from the model registry. This feature is in preview, and we would greatly appreciate any feedback through this form: https://databricks.sjc1.qualtrics.com/jfe/form/SV_1H6Ovx38zgCKAR0.
# MAGIC
# MAGIC ## Instructions:
# MAGIC 1. Run the notebook against a cluster with Databricks ML Runtime version 13.2.x-cpu, to best re-create the training environment.
# MAGIC 2. Add additional data processing on your loaded table to match the model schema if necessary (see the "Define input and output" section below).
# MAGIC 3. "Run All" the notebook.
# MAGIC 4. Note: If the `%pip` does not work for your model (i.e. it does not have a `requirements.txt` file logged), modify to use `%conda` if possible.

# COMMAND ----------

model_name = "mini_mlops_pipeline"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Recreation
# MAGIC Run the notebook against a cluster with Databricks ML Runtime version 13.2.x-cpu, to best re-create the training environment.. The cell below downloads the model artifacts associated with your model in the remote registry, which include `conda.yaml` and `requirements.txt` files. In this notebook, `pip` is used to reinstall dependencies by default.
# MAGIC
# MAGIC ### (Optional) Conda Instructions
# MAGIC Models logged with an MLflow client version earlier than 1.18.0 do not have a `requirements.txt` file. If you are using a Databricks ML runtime (versions 7.4-8.x), you can replace the `pip install` command below with the following lines to recreate your environment using `%conda` instead of `%pip`.
# MAGIC ```
# MAGIC conda_yml = os.path.join(local_path, "conda.yaml")
# MAGIC %conda env update -f $conda_yml
# MAGIC ```

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os

model_uri = f"models:/{model_name}/Production"
local_path = ModelsArtifactRepository(model_uri).download_artifacts("") # download model from remote registry

requirements_path = os.path.join(local_path, "requirements.txt")
if not os.path.exists(requirements_path):
  dbutils.fs.put("file:" + requirements_path, "", True)

# COMMAND ----------

# MAGIC %pip install -r $requirements_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define input and output
# MAGIC The table path assigned to`input_table_name` will be used for batch inference and the predictions will be saved to `output_table_path`. After the table has been loaded, you can perform additional data processing, such as renaming or removing columns, to ensure the model and table schema matches.

# COMMAND ----------

# redefining key variables here because %pip and %conda restarts the Python interpreter
model_name = "mini_mlops_pipeline"
input_table_name = "andre_catalog.default.white_wine"
output_table_path = "/FileStore/batch-inference/mini_mlops_pipeline"

# COMMAND ----------

# load table as a Spark DataFrame
table = spark.table(input_table_name)

# optionally, perform additional data processing (may be necessary to conform the schema)


# COMMAND ----------

if "quality" in table.columns:  ## Note: added
    table = table.drop("quality")

# COMMAND ----------

# MAGIC %md ## Load model and run inference
# MAGIC **Note**: If the model does not return double values, override `result_type` to the desired type.

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct

model_uri = f"models:/{model_name}/Production"

# create spark user-defined function for model prediction
predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")

# COMMAND ----------

output_df = table.withColumn("prediction", predict(struct(*table.columns)))

# COMMAND ----------

# MAGIC %md ## Save predictions
# MAGIC **The default output path on DBFS is accessible to everyone in this Workspace. If you want to limit access to the output you must change the path to a protected location.**
# MAGIC The cell below will save the output table to the specified FileStore path. `datetime.now()` is appended to the path to prevent overwriting the table in the event that this notebook is run in a batch inference job. To overwrite existing tables at the path, replace the cell below with:
# MAGIC ```python
# MAGIC output_df.write.mode("overwrite").save(output_table_path)
# MAGIC ```
# MAGIC
# MAGIC ### (Optional) Write predictions to Unity Catalog
# MAGIC If you have access to any UC catalogs, you can also save predictions to UC by specifying a table in the format `<catalog>.<database>.<table>`.
# MAGIC ```python
# MAGIC output_table = "" # Example: "ml.batch-inference.mini_mlops_pipeline"
# MAGIC output_df.write.saveAsTable(output_table)
# MAGIC ```

# COMMAND ----------

from datetime import datetime

# To write to a unity catalog table, see instructions above
output_df.write.save(f"{output_table_path}_{datetime.now().isoformat()}".replace(":", "."))

# COMMAND ----------

output_df.display()
