# Databricks notebook source
# MAGIC %md # Basic Sklearn MLflow train and predict with Model Registry
# MAGIC 
# MAGIC Synopsis
# MAGIC * Demonstrated basic use case of creating a registered model and retrieving the production version of the model.
# MAGIC 
# MAGIC Overview
# MAGIC * Trains 5 runs with a different `max_depth` parameter and saves the model as sklearn flavor.
# MAGIC * Create a new registered model.
# MAGIC * Finds best model (smallest RMSE metric) and adds it to the registered as a `production` stage. 
# MAGIC * Loads and scores a model with the new `models` URI: `models/my_registered_model/productions`.
# MAGIC * Loads and scores with both sklearn and pyfunc/UDF flavors.
# MAGIC 
# MAGIC Setup
# MAGIC * MLflow 1.9.0 or above is required.
# MAGIC * Use DBR ML 6.2 which comes with MLflow 1.9.0 pre-installed.
# MAGIC * For earlier DBR ML versions, attach the MLflow 1.9.0 library to your cluster.
# MAGIC 
# MAGIC Widgets:
# MAGIC * Registered Model Name.
# MAGIC * Sleep Time. Number of seconds to sleep after call to `update_model_version()`` and before call to `create_model_version()``.
# MAGIC   * If we don't sleep, the version might not yet be in READY state due to eventual consistency issues.
# MAGIC   * ERROR: INVALID_STATE: Model version andre_sklearn_registry_test version 1 has invalid status PENDING_REGISTRATION. Expected status is READY.

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

notebook_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
default_model_name = notebook_name.split("/")[-1]
dbutils.widgets.text("Registered Model Name", default_model_name) 
dbutils.widgets.text("Sleep TIme", "5") 
sleep_time = int(dbutils.widgets.get("Sleep TIme"))
model_name = dbutils.widgets.get("Registered Model Name")
model_name, sleep_time

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()
experiment_name = notebook_name
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
experiment_id, experiment_name

# COMMAND ----------

# MAGIC %md ### Get data

# COMMAND ----------

data_path = download_wine_file()

# COMMAND ----------

import pandas as pd
data = pd.read_csv(data_path)
display(data)

# COMMAND ----------

# MAGIC %md ### Training pipeline

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import mlflow.sklearn

# COMMAND ----------

train, test = train_test_split(data, test_size=0.30, random_state=2019)
train, test = train_test_split(data)
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# COMMAND ----------

def train(max_depth):
    with mlflow.start_run(run_name="reg_test") as run:
        run_id = run.info.run_uuid
        dt = DecisionTreeRegressor(max_depth=max_depth)
        dt.fit(train_x, train_y)
        predictions = dt.predict(test_x)
        mlflow.log_param("max_depth", max_depth)
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        mlflow.log_metric("rmse", rmse)
        print(f"{experiment_id} {run_id} {round(rmse,3)}")
        mlflow.sklearn.log_model(dt, "sklearn-model")

# COMMAND ----------

# MAGIC %md ### Delete existing runs

# COMMAND ----------

runs = client.list_run_infos(experiment_id)
for info in runs:
    client.delete_run(info.run_id)

# COMMAND ----------

# MAGIC %md ### Run training several times with different `max_depth` parameter

# COMMAND ----------

max_depths = [1,2,4,5,16]
for x in max_depths:
    train(x)

# COMMAND ----------

# MAGIC %md ### Find best run

# COMMAND ----------

best_run = client.search_runs(experiment_id,"", order_by=["metrics.rmse asc"], max_results=1)[0]
best_run

# COMMAND ----------

round(best_run.data.metrics['rmse'],3)

# COMMAND ----------

# MAGIC %md ## Model Registry

# COMMAND ----------

# MAGIC %md ### Create new registered model

# COMMAND ----------

delete_registered_model(client, model_name)

# COMMAND ----------

client.create_registered_model(model_name)
registered_model = client.get_registered_model(model_name)
registered_model.__dict__

# COMMAND ----------

display_registered_model_uri(model_name)

# COMMAND ----------

# MAGIC %md ### Create a new version - add a run to the registered model

# COMMAND ----------

best_run.info.artifact_uri

# COMMAND ----------

import time
source = f"{best_run.info.artifact_uri}/sklearn-model"
client.create_model_version(model_name, source, best_run.info.run_id)

# COMMAND ----------

# MAGIC %md ### Sleep a few seconds due to eventual consistency

# COMMAND ----------

time.sleep(sleep_time)

# COMMAND ----------

# MAGIC %md ### Label the version as `production` stage

# COMMAND ----------

#client.update_model_version(model_name, 1, stage="Production", description="My prod version") # 1.8.0
client.transition_model_version_stage (model_name, 1, "Production") # 1.9.0

# COMMAND ----------

client.get_latest_versions(model_name)

# COMMAND ----------

# MAGIC %md ## Predict
# MAGIC * Fetch the `production` model

# COMMAND ----------

 data_predict = data.drop(['quality'], axis=1)

# COMMAND ----------

model_uri = f"models:/{model_name}/production"
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as sklearn

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
model

# COMMAND ----------

predictions = model.predict(data_predict)
pdf = pd.DataFrame(predictions).head(5) 
pdf.columns = ['prediction']
display(pdf)

# COMMAND ----------

# MAGIC %md #### Predict as UDF

# COMMAND ----------

df = spark.createDataFrame(data_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns))
display(predictions)