# Databricks notebook source
# MAGIC %md # Notebook for Lightweight MLflow CICD example
# MAGIC 
# MAGIC **Overview**
# MAGIC * Notebook runs both as interactive and as a notebook task from job run
# MAGIC * Trains and saves model as sklearn
# MAGIC * Predicts using sklearn flavors
# MAGIC 
# MAGIC **Widgets**
# MAGIC * Experiment Name - If blank will use the default notebook experiment. The notebook experiment is not available when you run the notebook as a job, so we have to explicity call `mlflow.set_experiment()`.
# MAGIC * Max Depth - Parameter to model run
# MAGIC * Run Name - Name associated with the MLflow run.
# MAGIC * Scratch Dir - Folder where to copy the wine quality dataset.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

import os
import requests
import sklearn
import mlflow
import mlflow.sklearn
print("MLflow version:", mlflow.__version__)
print("Sklearn version:", sklearn.__version__)

# COMMAND ----------

dbutils.widgets.text("Experiment Name", "") 
dbutils.widgets.text("Max Depth", "1") 
dbutils.widgets.text("Run Name", "Interactive")
dbutils.widgets.text("Scratch Dir", "dbfs:/tmp/mlflow_cicd")

experiment_name = dbutils.widgets.get("Experiment Name")
max_depth = int(dbutils.widgets.get("Max Depth"))
run_name = dbutils.widgets.get("Run Name")
scratch_dir = dbutils.widgets.get("Scratch Dir")

output_file = os.path.join(scratch_dir, "mlflow_cicd.log")

print("experiment_name:", experiment_name)
print("run_name:", run_name)
print("scratch_dir:", scratch_dir)
print("output_file:", output_file)
print("max_depth:", max_depth)

dbutils.fs.mkdirs(scratch_dir)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
if experiment_name == "": # if running as notebook
    experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
else: # if running as job
    mlflow.set_experiment(experiment_name)
experiment = client.get_experiment_by_name(experiment_name)
print("experiment_id:", experiment.experiment_id)
print("experiment_name:", experiment.name)

# COMMAND ----------

colLabel = "quality"
colPrediction = "prediction"

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

def download_uri(uri, path):
    if os.path.exists(path):
        print(f"File {path} already exists")
    else:
        print(f"Downloading:\n  src: {uri}\n  dst: {path}")
        rsp = requests.get(uri)
        dbutils.fs.put(path, rsp.text, True)

WINE_URI = "https://raw.githubusercontent.com/amesar/mlflow-examples/master/data/wine-quality-white.csv"
data_path = os.path.join(scratch_dir, "wine-quality-white.csv")
download_uri(WINE_URI, data_path)

# COMMAND ----------

import pandas as pd
data = pd.read_csv(data_path.replace("dbfs:","/dbfs"))

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.30, random_state=42)
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# COMMAND ----------

with mlflow.start_run(run_name=run_name) as run:
    print("MLflow:")
    print("  run_id:",run.info.run_id)
    print("  experiment_id:",run.info.experiment_id)
    print("  max_depth:",max_depth)
    print("  scratch_dir:",scratch_dir)
    print("  data_path:",data_path)

    mlflow.log_param("max_depth", max_depth)
    mlflow.set_tag("run_id", run.info.run_id)
    mlflow.set_tag("experiment_id", experiment.experiment_id)
    mlflow.set_tag("experiment_name", experiment_name)
    mlflow.set_tag("scratch_dir", scratch_dir)
    mlflow.set_tag("data_path", data_path)

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(train_x, train_y)
    mlflow.sklearn.log_model(model, "sklearn-model")
    
    predictions = model.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    print("  rmse:",rmse)
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

# MAGIC %md ### Display Run URI

# COMMAND ----------

def get_tag(tag_name):
    try:
        return dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get(tag_name).get()
    except Exception as e: # py4j.protocol.Py4JJavaError
        return None

# COMMAND ----------

host_name = get_tag("browserHostName")
if host_name:
    uri = f"https://{host_name}/#mlflow/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

# MAGIC %md #### Write the run ID to a DBFS log file

# COMMAND ----------

dbutils.fs.put(output_file, run.info.run_id, True)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = f"runs:/{run.info.run_id}/sklearn-model"

# COMMAND ----------

# MAGIC %md #### Predict as sklearn

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
data_to_predict = data.drop(colLabel, axis=1)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[colPrediction]))

# COMMAND ----------

# MAGIC %md ### Check scratch directory

# COMMAND ----------

os.environ["SCRATCH_DIR"] = scratch_dir.replace("dbfs:","/dbfs")

# COMMAND ----------

# MAGIC %sh ls -l $SCRATCH_DIR