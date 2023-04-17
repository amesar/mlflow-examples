# Databricks notebook source
# MAGIC %md ## Sklearn Wine Quality MLflow model
# MAGIC * Trains and saves model as Sklearn flavor
# MAGIC * Predicts using Sklearn, PyFunc and UDF flavors
# MAGIC 
# MAGIC ### Widgets
# MAGIC * 1. Experiment name: if not set, use notebook experiment
# MAGIC * 2. Registered model: if not set, do not register as model
# MAGIC * 3. Model version stage: model stage
# MAGIC * 4. Archive existing versions: 
# MAGIC * 5. Save signature
# MAGIC * 6. SHAP
# MAGIC * 7. Delta table: if not set, read CSV file from internet
# MAGIC * 8. Max depth
# MAGIC * 9. Max leaf nodes
# MAGIC 
# MAGIC Last udpated: 2023-04-17 - Repo variant

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Experiment name","")
dbutils.widgets.text("2. Registered model","")
dbutils.widgets.dropdown("3. Model version stage","None", _model_version_stages)
dbutils.widgets.dropdown("4. Archive existing versions","no",["yes","no"])
dbutils.widgets.dropdown("5. Save signature","no",["yes","no"])
dbutils.widgets.dropdown("6. SHAP","no",["yes","no"])
dbutils.widgets.text("7. Delta table", "")
dbutils.widgets.text("8. Max depth", "1") 
dbutils.widgets.text("9. Max leaf nodes", "")

experiment_name = dbutils.widgets.get("1. Experiment name")
model_name = dbutils.widgets.get("2. Registered model")
model_version_stage = dbutils.widgets.get("3. Model version stage")
archive_existing_versions = dbutils.widgets.get("4. Archive existing versions") == "yes"
save_signature = dbutils.widgets.get("5. Save signature") == "yes"
shap = dbutils.widgets.get("6. SHAP") == "yes"
delta_table = dbutils.widgets.get("7. Delta table")
max_depth = to_int(dbutils.widgets.get("8. Max depth"))
max_leaf_nodes = to_int(dbutils.widgets.get("9. Max leaf nodes"))

if model_name=="": model_name = None
if model_version_stage=="None": model_version_stage = None
if experiment_name=="None": experiment_name = None

print("experiment_name:", experiment_name)
print("model_name:", model_name)
print("model_version_stage:", model_version_stage)
print("archive_existing_versions:", archive_existing_versions)
print("save_signature:", save_signature)
print("SHAP:", shap)
print("delta_table:", delta_table)
print("max_depth:", max_depth)
print("max_leaf_nodes:", max_leaf_nodes)

# COMMAND ----------

if experiment_name:
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    print("exp:",exp)
    client.set_experiment_tag(exp.experiment_id, "version_mlflow", mlflow.__version__)
    client.set_experiment_tag(exp.experiment_id, "timestamp", now)

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data = WineQuality.get_data(delta_table)
train_x, test_x, train_y, test_y = WineQuality.prep_training_data(data)
display(data)

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

# COMMAND ----------

import os, platform

run_name=f"{now} - {mlflow.__version__}" 
with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id
    print("MLflow:")
    print("  run_id:", run_id)
    print("  experiment_id:", run.info.experiment_id)
    print("Parameters:")
    print("  max_depth:", max_depth)
    print("  max_leaf_nodes:", max_leaf_nodes)
    
    mlflow.set_tag("mlflow.runName", run_id) # ignored unlike OSS MLflow
    mlflow.set_tag("timestamp", now)
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("version.sklearn", sklearn.__version__)
    mlflow.set_tag("version.DATABRICKS_RUNTIME_VERSION", os.environ.get("DATABRICKS_RUNTIME_VERSION",None))
    mlflow.set_tag("version.python", platform.python_version())
    mlflow.set_tag("save_signature", save_signature)

    model = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x, train_y)
      
    predictions = model.predict(test_x)
    signature = infer_signature(train_x, predictions) if save_signature else None
    print("signature:",signature)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
        
    mlflow.sklearn.log_model(model, "model", signature=signature)
    if model_name:
        version = register_model(run, model_name, model_version_stage)
        
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    r2 = r2_score(test_y, predictions)
    print("Metrics:")
    print("  rmse:",rmse)
    print("  r2:",r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2) 
    if shap:
        mlflow.shap.log_explanation(model.predict, train_x)

# COMMAND ----------

# Set run name to the run ID

print("Old runName:", run.data.tags.get("mlflow.runName",None))
client.set_tag(run_id, "mlflow.runName", run_id)
run = client.get_run(run_id)
print("New runName:", run.data.tags.get("mlflow.runName",None))

# COMMAND ----------

# MAGIC %md ### Display UI links

# COMMAND ----------

display_run_uri(run.info.experiment_id, run_id)

# COMMAND ----------

display_experiment_id_info(run.info.experiment_id)

# COMMAND ----------

if model_name:
    display_registered_model_version_uri(model_name, version.version)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as sklearn

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
data_to_predict = WineQuality.prep_prediction_data(data)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[WineQuality.colPrediction]))

# COMMAND ----------

type(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as PyFunc

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[WineQuality.colPrediction]))

# COMMAND ----------

type(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as Spark UDF

# COMMAND ----------

df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns)).select("prediction")
display(predictions)

# COMMAND ----------

type(predictions)
