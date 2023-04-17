# Databricks notebook source
# MAGIC %md ## Sklearn Iris MLflow model
# MAGIC 
# MAGIC Simple Sklearn model.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Experiment name","")
experiment_name = dbutils.widgets.get("1. Experiment name")

dbutils.widgets.text("2. Data path", "") 
data_path = dbutils.widgets.get("2. Data path")
if data_path=="": data_path = None

dbutils.widgets.text("3. Max depth", "1") 
max_depth = to_int(dbutils.widgets.get("3. Max depth"))

print("experiment_name:", experiment_name)
print("data_path:", data_path)
print("max_depth:", max_depth)

# COMMAND ----------

import mlflow

if experiment_name:
    mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md ### Get data

# COMMAND ----------

from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_data(data_path=None):
    if not data_path:
        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)
    else:
        import pandas as pd
        df = pd.read_csv(data_path)
        train, test = train_test_split(df, test_size=0.30, random_state=42)
        col_label = "species"
        X_train = train.drop([col_label], axis=1)
        X_test = test.drop([col_label], axis=1)
        y_train = train[[col_label]]
        y_test = test[[col_label]]
    return X_train, X_test, y_train, y_test

# COMMAND ----------

X_train, X_test, y_train, y_test = get_data(data_path)

# COMMAND ----------

# MAGIC %md ### Train model

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

run_name=f"{now} - {mlflow.__version__}" 
with mlflow.start_run(run_name=run_name) as run:
    print("run_id:", run.info.run_id)
    print("experiment_id:", run.info.experiment_id)
    mlflow.set_tag("mlflow_version", mlflow.__version__)
    mlflow.set_tag("data_path", data_path)
    mlflow.log_param("max_depth",max_depth)
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

# COMMAND ----------

# MAGIC %md ### Display UI links

# COMMAND ----------

display_run_uri(run.info.experiment_id, run.info.run_id)

# COMMAND ----------

display_experiment_id_info(run.info.experiment_id)
