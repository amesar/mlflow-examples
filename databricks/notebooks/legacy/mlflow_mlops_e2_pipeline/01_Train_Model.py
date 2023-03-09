# Databricks notebook source
# MAGIC %md # Train Sklearn model
# MAGIC 
# MAGIC **Overview**
# MAGIC * Trains a Sklearn model several times with different values for `max_depth` hyperparameter.
# MAGIC * Runs will be in the notebook experiment.
# MAGIC * Algorithm is DecisionTreeRegressor with wine quality dataset.
# MAGIC * Dataset: wine quality.
# MAGIC * Option to use [MLflow Autologging](https://docs.databricks.com/applications/mlflow/databricks-autologging.html) (default) or explicit MLflow API calls.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.dropdown("Autologging","yes",["yes","no"])
autologging = dbutils.widgets.get("Autologging") == "yes"
autologging

# COMMAND ----------

# MAGIC %md ### Delete any existing runs
# MAGIC * We want to start with a clean slate for demo purposes.
# MAGIC * In a real world case, you probably wouldn't do this.

# COMMAND ----------

delete_runs(experiment_id)

# COMMAND ----------

# MAGIC %md ## Prepare data

# COMMAND ----------

import pandas as pd
data = pd.read_csv(data_path)
display(data)

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.30, random_state=42)

# The predicted column is col_label which is a scalar from [3, 9]
train_x = train.drop([col_label], axis=1)
test_x = test.drop([col_label], axis=1)
train_y = train[col_label]
test_y = test[col_label]

# COMMAND ----------

# MAGIC %md ### Training Pipeline

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# COMMAND ----------

# MAGIC %md #### Train with explicit calls to MLflow API

# COMMAND ----------

def train_no_autologging(max_depth):
    import mlflow
    with mlflow.start_run(run_name="sklearn") as run:
        mlflow.log_param("max_depth", max_depth)
        mlflow.set_tag("version.mlflow", mlflow.__version__)
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(train_x, train_y)
        mlflow.sklearn.log_model(model, "model")     
        predictions = model.predict(test_x)
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        mlflow.log_metric("training_rmse", rmse)
        print(f"{rmse:5.3f} {max_depth:8d} {run.info.run_id}")

# COMMAND ----------

# MAGIC %md #### Train with MLflow Autologging

# COMMAND ----------

def train_with_autologging(max_depth):
    dt = DecisionTreeRegressor(max_depth=max_depth)
    dt.fit(train_x, train_y)
    predictions = dt.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    print(f"{rmse:5.3f} {max_depth:2d}")

# COMMAND ----------

# MAGIC %md ### Train with different `max_depth` hyperparameter values

# COMMAND ----------

if autologging:
    print("RMSE   MaxDepth")
else:
    print("RMSE  MaxDepth Run ID")
    
max_depths = [1, 2, 4, 8, 16]
for max_depth in max_depths:
    if autologging:
        train_with_autologging(max_depth)
    else:
        train_no_autologging(max_depth)