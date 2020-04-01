# Databricks notebook source
# MAGIC %md # Sklearn MLflow train and predict for CICD
# MAGIC * Trains and saves model as sklearn
# MAGIC * Predicts using sklearn and pyfunc UDF flavors
# MAGIC * https://demo.cloud.databricks.com/#mlflow/experiments/6231308

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

import sklearn
import mlflow
import mlflow.sklearn
print("MLflow Version:", mlflow.__version__)
print("sklearn version:",sklearn.__version__)

# COMMAND ----------

#dbutils.widgets.remove("Experiment Name")

# COMMAND ----------

##default_experiment_name = get_experiment_name()

dbutils.widgets.text("Experiment Name", "") 
dbutils.widgets.text("Max Depth", "1") 
dbutils.widgets.text("Run Name", "Local")
dbutils.widgets.text("Output", "dbfs:/tmp/mlflow_cicd_test.log")

experiment_name = dbutils.widgets.get("Experiment Name")
max_depth = int(dbutils.widgets.get("Max Depth"))
run_name = dbutils.widgets.get("Run Name")
output_file = dbutils.widgets.get("Output")

print("experiment_name:",experiment_name)
print("run_name:",run_name)
print("output_file:",output_file)
print("max_depth:",max_depth)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
if experiment_name == "":
    experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
else:
    mlflow.set_experiment(experiment_name)
experiment = client.get_experiment_by_name(experiment_name)
print("experiment_id:",experiment.experiment_id)
print("experiment_name:",experiment.name)

# COMMAND ----------

colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data_path = download_wine_file()

# COMMAND ----------

import pandas as pd
data = pd.read_csv(data_path)

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.30, random_state=2019)
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import pandas as pd
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
    mlflow.log_param("max_depth", max_depth)
    mlflow.set_tag("run_name", run_name)
    mlflow.set_tag("run_id", run.info.run_id)
    mlflow.set_tag("experiment_id", experiment.experiment_id)
    mlflow.set_tag("experiment_name", experiment_name)

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(train_x, train_y)
    mlflow.sklearn.log_model(model, "sklearn-model")
    
    predictions = model.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    print("  rmse:",rmse)
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

# MAGIC %md #### Write the run ID in DBFS log file

# COMMAND ----------

dbutils.fs.put(output_file, run.info.run_id, True)

# COMMAND ----------

#display_run_uri(run.info.experiment_id, run.info.run_id)
# https://demo.cloud.databricks.com/#mlflow/experiments/6230591/runs/e5e4879ed7c447ffaa72dc7fec0833c6

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)

# COMMAND ----------

# MAGIC %md #### Predict as sklearn

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
data_to_predict = data.drop(colLabel, axis=1)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[colPrediction]))

# COMMAND ----------

# MAGIC %scala
# MAGIC dbutils.notebook.getContext.tags.mkString("\n")

# COMMAND ----------

