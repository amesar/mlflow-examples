# Databricks notebook source
# MAGIC %md # Basic Sklearn MLflow train and predict with Databricks model server
# MAGIC 
# MAGIC **Overview**
# MAGIC * End-to-end example of train a model, deploy it to Databricks model server and score.
# MAGIC * Trains and registers model.
# MAGIC * Deploys registered model to Databricks model server.
# MAGIC * Client invocation: curl and Python `requests` examples.
# MAGIC 
# MAGIC **Widgets**
# MAGIC * Registered model: if left empty, the registered model name will be created from your user name plus the notebook name.
# MAGIC   * For example, `john.doe@databricks` will result in `john_doe_02e_Sklearn_Train_Predict_ModelServer`.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("Registered Model","")

default_registered_model = ""
registered_model = dbutils.widgets.get("Registered Model")
if registered_model == "":
    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    username = context.tags().get("user").get().replace("@databricks.com","").replace(".","_")
    notebook = context.notebookPath().get().split("/")[-1]
    registered_model = f"{username}_{notebook}"

print("default_registered_model:",default_registered_model)
print("registered_model:",registered_model)

if not registered_model :
    raise Exception("Required value for registered_model")

# COMMAND ----------

import sklearn
import mlflow
import mlflow.sklearn

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data_path = download_wine_file()

# COMMAND ----------

from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv(data_path)

train, test = train_test_split(data, test_size=0.30, random_state=42)

train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

model_name = "sklearn-model"
max_depth = 5

with mlflow.start_run(run_name="sklearn") as run:
    run_id = run.info.run_id
    print("run_id:",run_id)
    print("experiment_id:",run.info.experiment_id)
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.log_param("max_depth", max_depth)
    
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    
    mlflow.sklearn.log_model(model, "sklearn-model", registered_model_name=registered_model)
        
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

display_run_uri(run.info.experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Get latest version

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model = client.get_registered_model(registered_model)
versions = [vr.version for vr in model.latest_versions]
versions.sort()
latest_version = versions[-1]
print("Latest mode version:",latest_version)

# COMMAND ----------

# MAGIC %md ### Deploy model - manual step
# MAGIC 
# MAGIC * Now you must start the model server for the version you have just created.
# MAGIC * Follow the steps in [MLflow Model Serving on Databricks](https://docs.databricks.com/applications/mlflow/model-serving.html) to deploy your model to the Databricks Model Server.

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

display_registered_model_uri(registered_model)

# COMMAND ----------

# Wait for the model version to be deployed to existing model server
import time
time.sleep(10)

# COMMAND ----------

# MAGIC %md #### Setup for prediction

# COMMAND ----------

hostname = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
model_server_uri= f"https://{host_name}/model/{registered_model}/{latest_version}/invocations"
os.environ["MODEL_SERVER_URI"] = model_server_uri

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["TOKEN"] = token

model_server_uri

# COMMAND ----------

# MAGIC %md #### Predict with curl

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC echo "MODEL_SERVER_URI: $MODEL_SERVER_URI"
# MAGIC 
# MAGIC curl -X POST -H "Content-Type:application/json" \
# MAGIC   -H "Authorization: Bearer $TOKEN" \
# MAGIC   -d'{
# MAGIC   "columns": [
# MAGIC     "fixed acidity",
# MAGIC     "volatile acidity",
# MAGIC     "citric acid",
# MAGIC     "residual sugar",
# MAGIC     "chlorides",
# MAGIC     "free sulfur dioxide",
# MAGIC     "total sulfur dioxide",
# MAGIC     "density",
# MAGIC     "pH",
# MAGIC     "sulphates",
# MAGIC     "alcohol"
# MAGIC   ],
# MAGIC   "data": [
# MAGIC     [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8 ],
# MAGIC     [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5 ],
# MAGIC     [ 8.1, 0.28, 0.4,   6.9, 0.05,  30,  97, 0.9951, 3.26, 0.44, 10.1 ]
# MAGIC   ]
# MAGIC }' \
# MAGIC   $MODEL_SERVER_URI

# COMMAND ----------

# MAGIC %md #### Predict with Python

# COMMAND ----------

data = """
{
  "columns": [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
  ],
  "data": [
    [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8 ],
    [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5 ],
    [ 8.1, 0.28, 0.4,   6.9, 0.05,  30,  97, 0.9951, 3.26, 0.44, 10.1 ]
  ]
}
"""

# COMMAND ----------

import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
response = requests.request(method='POST', headers=headers, url=model_server_uri, data=data)
response.json()