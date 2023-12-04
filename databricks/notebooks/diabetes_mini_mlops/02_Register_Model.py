# Databricks notebook source
# MAGIC %md ## Register Best Model Run in Model Registry
# MAGIC * Creates a new registered model if it doesn't already exist.
# MAGIC * Deletes all current model versions (optional).
# MAGIC * Finds the best model (lowest 'rmse' metric) generated from [01_Train_Model]($01_Train_Model) notebook experiment.
# MAGIC * Adds the best run as a registered model version and promotes it to the `production` stage.
# MAGIC
# MAGIC ##### Notes
# MAGIC * andre_catalog.ml_models2.diabetes_mlops

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
model_name = dbutils.widgets.get("1. Registered model")

dbutils.widgets.text("2. Metric", "rmse")
metric = dbutils.widgets.get("2. Metric")

# COMMAND ----------

assert_widget(model_name, "1. Registered model")
assert_widget(metric, "2. Metric")

# COMMAND ----------

experiment = mlflow_client.get_experiment_by_name(_experiment_name)
experiment_id = experiment.experiment_id
display_experiment_uri(experiment)

# COMMAND ----------

# MAGIC %md ### Show all the RMSE metrics and max_depth params of all runs

# COMMAND ----------

df = mlflow.search_runs(experiment_id, order_by=[f"metrics.{metric} ASC"])
df = df[["run_id",f"metrics.{metric}","params.l1_ratio"]]
df = df.round({f"metrics.{metric}": 3})
display(df)

# COMMAND ----------

# MAGIC %md ### Find best run
# MAGIC
# MAGIC Search for the run with lowest training RMSE metric.

# COMMAND ----------

runs = mlflow_client.search_runs(experiment_id, order_by=[f"metrics.{metric} ASC"], max_results=1)
best_run = runs[0]

print("Run ID:     ", best_run.info.run_id)
print("Best metric:", round(best_run.data.metrics[metric],3))
print("Best params:", best_run.data.params)

# COMMAND ----------

# MAGIC %md ### Create registered model 
# MAGIC
# MAGIC If the model already exists remove, then delete all existing versions.

# COMMAND ----------

create_registered_model(model_name)

# COMMAND ----------

# MAGIC %md ### Create model version for best run

# COMMAND ----------

source = f"{best_run.info.artifact_uri}/model"
source

# COMMAND ----------

model_name

# COMMAND ----------

version = mlflow_client.create_model_version(
    name = model_name, 
    source = source, 
    run_id = best_run.info.run_id
)

# COMMAND ----------

# MAGIC %md ### Create model alias for best run

# COMMAND ----------

_alias

# COMMAND ----------

mlflow_client.set_registered_model_alias(model_name, _alias, version.version)

# COMMAND ----------

# MAGIC %md ### Display version

# COMMAND ----------

version = mlflow_client.get_model_version(version.name, version.version)
dump_obj(version)
