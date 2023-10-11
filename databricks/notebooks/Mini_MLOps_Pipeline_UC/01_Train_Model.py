# Databricks notebook source
# MAGIC %md # Train Sklearn Model
# MAGIC
# MAGIC **Overview**
# MAGIC * Option to use [MLflow Autologging](https://docs.databricks.com/applications/mlflow/databricks-autologging.html).
# MAGIC * Train a Sklearn model several times with different `maxDepth` hyperparameter values.
# MAGIC   * Runs will be in the notebook experiment.
# MAGIC * Algorithm: DecisionTreeRegressor.
# MAGIC
# MAGIC **Widgets**
# MAGIC   * Autologging - yes or no.
# MAGIC   * Delete existing runs - Deletes existing experiment runs before executing training runs.
# MAGIC   * Delta table - If specified will read data from a Delta table. The table will be created if it doesn't already exist. Otherwise  read wine quality data via HTTP download.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

dbutils.widgets.dropdown("1. Autologging","no",["yes","no"])
autologging = dbutils.widgets.get("1. Autologging") == "yes"

dbutils.widgets.dropdown("2. Delete existing runs","yes",["yes","no"])
do_delete_runs = dbutils.widgets.get("2. Delete existing runs") == "yes"

dbutils.widgets.text("3. Delta table", "")
delta_table = dbutils.widgets.get("3. Delta table")

autologging, do_delete_runs
print("autologging:", autologging)
print("do_delete_runs:", do_delete_runs)
print("delta_table:", delta_table)

# COMMAND ----------

experiment = init()

# COMMAND ----------

# MAGIC %md ### Delete any existing runs

# COMMAND ----------

if do_delete_runs:
    delete_runs(experiment.experiment_id)

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data = get_wine_quality_data(delta_table)
display(data)

# COMMAND ----------

data.describe()

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.30, random_state=42)
X_train = train.drop([_col_label], axis=1)
X_test = test.drop([_col_label], axis=1)
y_train = train[[_col_label]]
y_test = test[[_col_label]]

# COMMAND ----------

# MAGIC %md ### Training Pipeline

# COMMAND ----------

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# COMMAND ----------

# MAGIC %md #### Train with explicit calls to MLflow API

# COMMAND ----------

from mlflow.models.signature import infer_signature

# COMMAND ----------


def train_no_autologging(max_depth):
    import time
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
    with mlflow.start_run(run_name=f"No_autolog") as run:
        mlflow.set_tag("mlflow_version", mlflow.__version__)
        mlflow.set_tag("experiment_name", experiment.name)
        mlflow.log_param("max_depth", max_depth)
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mlflow.log_metric("training_rmse", rmse)
        print(f"rmse={rmse:5.3f} max_depth={max_depth:02d} run_id={run.info.run_id}")
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(model, "model", signature=signature)

# COMMAND ----------

# MAGIC %md #### Train with MLflow Autologging

# COMMAND ----------

def train_with_autologging(max_depth):
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"rmse={rmse:5.3f} max_depth={max_depth:02d}")

# COMMAND ----------

# MAGIC %md ### Train model with different hyperparameters
# MAGIC * In a realistic scenario you would use hyperopt or similar. See:
# MAGIC     * [Hyperopt concepts](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html)
# MAGIC   * [Hyperparameter tuning](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/index.html)

# COMMAND ----------

max_depths = [1, 2, 4, 8, 16]
#max_depths = [1]

for max_depth in max_depths:
    if autologging:
        train_with_autologging(max_depth)
    else:
        train_no_autologging(max_depth)

# COMMAND ----------

display_experiment_uri(experiment)

# COMMAND ----------

# MAGIC %md ### Next notebook
# MAGIC
# MAGIC Next go to the **[02_Register_Model]($02_Register_Model)** notebook.
