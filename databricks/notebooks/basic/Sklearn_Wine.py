# Databricks notebook source
# MAGIC %md ## Sklearn Wine Quality MLflow model
# MAGIC * Trains and saves model as Sklearn flavor
# MAGIC * Optionally registers the run as new registered model
# MAGIC * Predicts using Sklearn, Pyfunc and UDF flavors
# MAGIC * Works with either Workspace or Unity Catalog model registry
# MAGIC * See [Sklearn_Wine_UC]($Sklearn_Wine_UC) a Unity Catalog specific
# MAGIC
# MAGIC ### Widgets
# MAGIC * 01. Run name
# MAGIC * 02. Experiment name: if not set, use notebook experiment
# MAGIC * 03. Registered model: if set, register as model
# MAGIC * 04. Model version stage
# MAGIC * 05. Archive existing versions
# MAGIC * 06. Model alias
# MAGIC * 07. Save signature
# MAGIC * 08. Input example
# MAGIC * 09. Log input - new in MLflow 2.4.0
# MAGIC * 10. Log evaluation metric
# MAGIC * 11. Log SHAP
# MAGIC * 12. Delta table: if not set, read default Databricks CSV file from DBFS
# MAGIC * 13. Max depth
# MAGIC
# MAGIC #### Notes
# MAGIC
# MAGIC * Registered model:
# MAGIC   * Sklearn_Wine_test
# MAGIC   
# MAGIC * Experiment:
# MAGIC   * /Users/me@databricks.com/experiments/sklearn_wine/Sklearn_Wine_ws
# MAGIC   * /Users/me@databricks.com/experiments/best/Sklearn_Wine_repo
# MAGIC
# MAGIC * Delta tables: 
# MAGIC   * andre.wine_quality
# MAGIC   * andre_catalog.ml_data.winequality_white
# MAGIC   * andre_catalog.ml_data.winequality_red
# MAGIC
# MAGIC Last udpated: _2024-04-20_

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("01. Run name", "")
dbutils.widgets.text("02. Experiment name", "")
dbutils.widgets.text("03. Registered model", "Sklearn_wine_best")
dbutils.widgets.dropdown("04. Model version stage", "None", _model_version_stages)
dbutils.widgets.dropdown("05. Archive existing versions", "no", ["yes","no"])
dbutils.widgets.text("06. Model alias","")
dbutils.widgets.dropdown("07. Save signature", "yes", ["yes","no"])
dbutils.widgets.dropdown("08. Input example", "yes", ["yes","no"])
dbutils.widgets.dropdown("09. Log input", "yes", ["yes","no"])
dbutils.widgets.dropdown("10. Log evaluation metrics", "no", ["yes","no"])
dbutils.widgets.dropdown("11. Log SHAP", "no", ["yes","no"])
dbutils.widgets.text("12. Delta table", "andre_catalog.ml_data.winequality_white")
dbutils.widgets.text("13. Max depth", "1") 

run_name = dbutils.widgets.get("01. Run name")
experiment_name = dbutils.widgets.get("02. Experiment name")
model_name = dbutils.widgets.get("03. Registered model")
model_version_stage = dbutils.widgets.get("04. Model version stage")
archive_existing_versions = dbutils.widgets.get("05. Archive existing versions") == "yes"
model_alias = dbutils.widgets.get("06. Model alias")
save_signature = dbutils.widgets.get("07. Save signature") == "yes"
input_example = dbutils.widgets.get("08. Input example") == "yes"
log_input = dbutils.widgets.get("09. Log input") == "yes"
log_evaluation_metrics = dbutils.widgets.get("10. Log evaluation metrics") == "yes"
log_shap = dbutils.widgets.get("11. Log SHAP") == "yes"
delta_table = dbutils.widgets.get("12. Delta table")
max_depth = to_int(dbutils.widgets.get("13. Max depth"))

run_name = run_name or None
experiment_name = experiment_name or None
model_name = model_name or None
model_version_stage = model_version_stage or None
model_alias = model_alias or None
input_example = input_example or None

toggle_unity_catalog(model_name)

print("\nrun_name:", run_name)
print("experiment_name:", experiment_name)
print("model_name:", model_name)
print("model_version_stage:", model_version_stage)
print("archive_existing_versions:", archive_existing_versions)
print("model_alias:", model_alias)
print("save_signature:", save_signature)
print("input_example:", input_example)
print("log_input:", log_input)
print("log_evaluation_metrics:", log_evaluation_metrics)
print("log_shap:", log_shap)
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

df_data, data_source = WineQuality.get_data(delta_table)
data_source

# COMMAND ----------

pdf_data =  df_data.toPandas()
display(pdf_data.head(10))

# COMMAND ----------

X_train, X_test, y_train, y_test = WineQuality.prep_training_data(pdf_data)

# COMMAND ----------

# MAGIC %md ### Set run name

# COMMAND ----------

def set_run_name_to_current_time(run_name):
    if run_name:
        return run_name
    else:
        return f"{now} - {mlflow.__version__}" 
_run_name = set_run_name_to_current_time(run_name)
run_name, _run_name

# COMMAND ----------

def set_run_name_to_run_id(run):
    print("Old runName:", run.data.tags.get("mlflow.runName"))
    client.set_tag(run_id, "mlflow.runName", run_id)
    run = client.get_run(run_id)
    print("New runName:", run.data.tags.get("mlflow.runName"))

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

# COMMAND ----------

import os, platform

with mlflow.start_run(run_name=_run_name) as run:
    run_id = run.info.run_id
    print("MLflow:")
    print("  run_id:", run_id)
    print("  experiment_id:", run.info.experiment_id)
    print("Parameters:")
    print("  max_depth:", max_depth)
    
    mlflow.set_tag("run_name", _run_name)
    mlflow.set_tag("timestamp", now)
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("version.sklearn", sklearn.__version__)
    mlflow.set_tag("version.DATABRICKS_RUNTIME_VERSION", os.environ.get("DATABRICKS_RUNTIME_VERSION",None))
    mlflow.set_tag("version.python", platform.python_version())
    mlflow.set_tag("save_signature", save_signature)
    mlflow.set_tag("input_example", input_example)
    mlflow.set_tag("log_input", log_input)
    mlflow.set_tag("data_source", data_source)
    mlflow.set_tag("workspace", databricks_utils.get_workspace_url())

    mlflow.log_param("max_depth", max_depth)

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    mlflow.set_tag("algorithm", type(model))
      
    predictions = model.predict(X_test)

    signature = infer_signature(X_train, predictions) if save_signature else None
    print("signature:", signature)
    print("input_example:", input_example)

    log_data_input(run, log_input, data_source, X_train)

    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("Metrics:")
    print("  rmse:",rmse)
    print("  r2:",r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2) 

    if log_evaluation_metrics:
        model_uri = mlflow.get_artifact_uri("model")
        print("model_uri:",model_uri)
        test_data = pd.concat([X_test, y_test], axis=1)
        result = mlflow.evaluate(
            model_uri,
            test_data,
            targets = "quality",
            model_type ="regressor",
            evaluators = "default",
            feature_names = list(pdf_data.columns),
            evaluator_config={"explainability_nsamples": 1000},
        )

    if log_shap:
        mlflow.shap.log_explanation(model.predict, X_train)

# COMMAND ----------

if not run_name:
    set_run_name_to_run_id(run)

# COMMAND ----------

# MAGIC %md ### Register model

# COMMAND ----------

if model_name:
    version = register_model(run, 
        model_name, 
        model_version_stage, 
        archive_existing_versions, 
        model_alias
    )
    print(f"Registered model '{model_name}' as version {version.version}")

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

# MAGIC %md ### Show input data

# COMMAND ----------

run = client.get_run(run_id)
if hasattr(run, "inputs") and run.inputs:
    for input in run.inputs:
        print(input)

# COMMAND ----------

run_id, run.info.run_id

# COMMAND ----------

# MAGIC %md ### Predict with `runs:/` URI

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as sklearn

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
data_to_predict = WineQuality.prep_prediction_data(pdf_data)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions, columns=[WineQuality.colPrediction]))

# COMMAND ----------

type(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as Pyfunc

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

# COMMAND ----------

# MAGIC %md ### Predict with `models:/` URI

# COMMAND ----------

if model_name:  
    model_uri = f"models:/{model_name}/{version.version}"
    print(model_uri)
else:
    print("No registered model specified")

# COMMAND ----------

# MAGIC %md #### Predict as Pyfunc

# COMMAND ----------

if model_name: 
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(data_to_predict)
    display(pd.DataFrame(predictions,columns=[WineQuality.colPrediction]))

# COMMAND ----------

# MAGIC %md #### Predict as Spark UDF

# COMMAND ----------

if model_name: 
    df = spark.createDataFrame(data_to_predict)
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    predictions = df.withColumn("prediction", udf(*df.columns)).select("prediction")
    display(predictions)

# COMMAND ----------

# MAGIC %md ### Predict with `models:/` URI and alias

# COMMAND ----------

if model_alias:
    model_uri = f"models:/{model_name}@{model_alias}"
    print("model_uri:", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(data_to_predict)
    display(pd.DataFrame(predictions,columns=[WineQuality.colPrediction]))
else:
    print("No model alias")
