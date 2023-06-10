# Databricks notebook source
# MAGIC %md ## Sklearn Wine Quality MLflow model
# MAGIC * Trains and saves model as Sklearn flavor
# MAGIC * Predicts using Sklearn, PyFunc and UDF flavors
# MAGIC
# MAGIC ### Widgets
# MAGIC * 01. Experiment name: if not set, use notebook experiment
# MAGIC * 02. Registered model: if set, register as model
# MAGIC * 03. Model version stage
# MAGIC * 04. Archive existing versions
# MAGIC * 05. Model alias
# MAGIC * 06. Save signature
# MAGIC * 07. Input example
# MAGIC * 08. Log input - MLflow 2.4.0
# MAGIC * 09. SHAP
# MAGIC * 10. Delta table: if not set, read CSV file from DBFS
# MAGIC * 11. Max depth
# MAGIC
# MAGIC Last udpated: 2023-06-09

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("01. Experiment name", "")
dbutils.widgets.text("02. Registered model", "")
dbutils.widgets.dropdown("03. Model version stage", "None", _model_version_stages)
dbutils.widgets.dropdown("04. Archive existing versions", "no", ["yes","no"])
dbutils.widgets.text("05. Model alias","")
dbutils.widgets.dropdown("06. Save signature", "no", ["yes","no"])
dbutils.widgets.dropdown("07. Input example", "no", ["yes","no"])
dbutils.widgets.dropdown("08. Log input", "no", ["yes","no"])
dbutils.widgets.dropdown("09. SHAP","no", ["yes","no"])
dbutils.widgets.text("10. Delta table", "")
dbutils.widgets.text("11. Max depth", "1") 

experiment_name = dbutils.widgets.get("01. Experiment name")
model_name = dbutils.widgets.get("02. Registered model")
model_version_stage = dbutils.widgets.get("03. Model version stage")
archive_existing_versions = dbutils.widgets.get("04. Archive existing versions") == "yes"
model_alias = dbutils.widgets.get("05. Model alias")
save_signature = dbutils.widgets.get("06. Save signature") == "yes"
input_example = dbutils.widgets.get("07. Input example") == "yes"
log_input = dbutils.widgets.get("08. Log input") == "yes"
shap = dbutils.widgets.get("09. SHAP") == "yes"
delta_table = dbutils.widgets.get("10. Delta table")
max_depth = to_int(dbutils.widgets.get("11. Max depth"))

model_name = model_name or None
model_version_stage = model_version_stage or None
model_alias = model_alias or None
experiment_name = experiment_name or None
input_example = input_example or None

print("experiment_name:", experiment_name)
print("model_name:", model_name)
print("model_version_stage:", model_version_stage)
print("archive_existing_versions:", archive_existing_versions)
print("model_alias:", model_alias)
print("save_signature:", save_signature)
print("input_example:", input_example)
print("log_input:", log_input)
print("SHAP:", shap)
print("delta_table:", delta_table)
print("max_depth:", max_depth)

# COMMAND ----------

if experiment_name:
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    print("Experiment ID:", exp.experiment_id)
    client.set_experiment_tag(exp.experiment_id, "version_mlflow", mlflow.__version__)
    client.set_experiment_tag(exp.experiment_id, "timestamp", now)

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

df, data_source = WineQuality.get_data(delta_table)
data_source

# COMMAND ----------

data =  df.toPandas()
display(data)

# COMMAND ----------

train_x, test_x, train_y, test_y = WineQuality.prep_training_data(data)

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
    
    mlflow.set_tag("mlflow.runName", run_id) # ignored unlike OSS MLflow
    mlflow.set_tag("timestamp", now)
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("version.sklearn", sklearn.__version__)
    mlflow.set_tag("version.DATABRICKS_RUNTIME_VERSION", os.environ.get("DATABRICKS_RUNTIME_VERSION",None))
    mlflow.set_tag("version.python", platform.python_version())
    mlflow.set_tag("save_signature", save_signature)
    mlflow.set_tag("input_example", input_example)
    mlflow.set_tag("log_input", log_input)
    mlflow.set_tag("data_source", data_source)

    mlflow.log_param("max_depth", max_depth)

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(train_x, train_y)
    mlflow.set_tag("algorithm", type(model))
      
    predictions = model.predict(test_x)

    signature = infer_signature(train_x, predictions) if save_signature else None
    print("signature:", signature)
    print("input_example:", input_example)

    # new in MLflow 2.4.0
    log_data_input(run, log_input, data_source, train_x)

    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=test_x)
    if model_name:
        version = register_model(run, 
            model_name, 
            model_version_stage, 
            archive_existing_versions, 
            model_alias
        )

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

print("Old runName:", run.data.tags.get("mlflow.runName"))
client.set_tag(run_id, "mlflow.runName", run_id)
run = client.get_run(run_id)
print("New runName:", run.data.tags.get("mlflow.runName"))

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

# MAGIC %md ### Show input data - new in MLflow 2.4.0

# COMMAND ----------

run = client.get_run(run_id)
if hasattr(run, "inputs") and run.inputs:
    for input in run.inputs:
        print(input)

# COMMAND ----------

run_id, run.info.run_id

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
display(pd.DataFrame(predictions, columns=[WineQuality.colPrediction]))

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
